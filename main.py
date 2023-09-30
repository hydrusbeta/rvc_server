import argparse
import base64
import json
import os.path
import subprocess
import traceback

import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common import *
from hay_say_common.cache import Stage
from jsonschema import ValidationError

ARCHITECTURE_NAME = 'rvc'
ARCHITECTURE_ROOT = os.path.join(ROOT_DIR, ARCHITECTURE_NAME)
INPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'input')
OUTPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'output')
WEIGHTS_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'weights')
INDICES_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'logs')
TEMP_FILE_EXTENSION = '.flac'

PYTHON_EXECUTABLE = os.path.join(ROOT_DIR, '.venvs', ARCHITECTURE_NAME, 'bin', 'python')
INFERENCE_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, 'command_line_interface.py')

INDEX_FILE_EXTENSION = '.index'
WEIGHTS_FILE_EXTENSION = '.pth'

app = Flask(__name__)


def register_methods(cache):
    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius, rms_mix_ratio, \
                protect, output_filename_sans_extension, gpu_id, session_id = parse_inputs()
            link_model_path(character)
            copy_input_audio(input_filename_sans_extension, session_id)
            execute_program(character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius,
                            rms_mix_ratio, protect, gpu_id, output_filename_sans_extension)
            copy_output(output_filename_sans_extension, session_id)
            clean_up(get_temp_files())
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = construct_full_error_message(INPUT_COPY_FOLDER, get_temp_files())

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    def parse_inputs():
        schema = {
            'type': 'object',
            'properties': {
                'Inputs': {
                    'type': 'object',
                    'properties': {
                        'User Audio': {'type': 'string'}
                    },
                    'required': ['User Audio']
                },
                'Options': {
                    'type': 'object',
                    'properties': {
                        'Character': {'type': 'string'},
                        'Pitch Shift': {'type': 'integer'},
                        'f0 Extraction Method': {'enum': ['crepe', 'harvest', 'parselmouth']},
                        'Index Ratio': {'type': 'number', 'minimum': 0, 'maximum': 1},
                        'Filter Radius': {'type': 'integer', 'minimum': 0},
                        'Voice Envelepe Mix Ratio': {'type': 'number', 'minimum': 0, 'maximum': 1},
                        'Voiceless Consonants Protection Ratio': {'type': 'number', 'minimum': 0, 'maximum': 0.5},
                    },
                    'required': ['Character', 'Pitch Shift', 'f0 Extraction Method', 'Index Ratio',
                                 'Voice Envelope Mix Ratio', 'Voiceless Consonants Protection Ratio']
                },
                'Output File': {'type': 'string'},
                'GPU ID': {'type': ['string', 'integer']},
                'Session ID' : {'type': ['string', 'null']}
            },
            'required': ['Inputs', 'Options', 'Output File', 'GPU ID', 'Session ID']
        }

        try:
            jsonschema.validate(instance=request.json, schema=schema)
        except ValidationError as e:
            raise BadInputException(e.message)

        input_filename_sans_extension = request.json['Inputs']['User Audio']
        character = request.json['Options']['Character']
        pitch_shift = request.json['Options']['Pitch Shift']
        f0_method = request.json['Options']['f0 Extraction Method']
        index_ratio = request.json['Options']['Index Ratio']
        filter_radius = request.json['Options'].get('Filter Radius')
        rms_mix_ratio = request.json['Options']['Voice Envelope Mix Ratio']
        protect = request.json['Options']['Voiceless Consonants Protection Ratio']
        output_filename_sans_extension = request.json['Output File']
        gpu_id = request.json['GPU ID']
        session_id = request.json['Session ID']

        return character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius, \
            rms_mix_ratio, protect, output_filename_sans_extension, gpu_id, session_id


    class BadInputException(Exception):
        pass


    def link_model_path(character):
        """Create a symbolic link to the model file in the location where RVC expects to find it."""
        symlink_file = os.path.join(WEIGHTS_FOLDER, character + WEIGHTS_FILE_EXTENSION)
        character_dir = get_model_path(ARCHITECTURE_NAME, character)
        weight_file = get_single_file_with_extension(character_dir, WEIGHTS_FILE_EXTENSION)
        create_link(weight_file, symlink_file)


    def copy_input_audio(input_filename_sans_extension, session_id):
        data, sr = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_filename_sans_extension)
        target = os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + TEMP_FILE_EXTENSION)
        try:
            soundfile.write(target, data, sr)
        except Exception as e:
            raise Exception("Unable to copy file from Hay Say's audio cache to rvc's raw directory.") from e


    # Hay Say displays the option 'parselmouth', but RVC expects 'pm'.
    # Map display labels to command line options using this dictionary.
    f0_command_line_option_from_display_option = {'crepe': 'crepe',
                                                  'harvest': 'harvest',
                                                  'parselmouth': 'pm'}


    def execute_program(character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius,
                        rms_mix_ratio, protect, gpu_id, output_filename_sans_extension):
        index_path = get_index_path(character)
        arguments = [
            '--voice', character + WEIGHTS_FILE_EXTENSION,
            '--input_filepath', os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + TEMP_FILE_EXTENSION),
            '--output_filepath', os.path.join(OUTPUT_COPY_FOLDER, output_filename_sans_extension + TEMP_FILE_EXTENSION),
            # Optional Parameters
            '--sid', str(0),
            '--transpose', str(pitch_shift),
            '--f0_method', f0_command_line_option_from_display_option[f0_method],
            *(['--index_filepath', index_path] if index_path else [None, None]),
            '--index_ratio', str(index_ratio),
            *(['--filter_radius', str(filter_radius)] if filter_radius else [None, None]),
            '--resample_rate', str(0),
            '--rms_mix_ratio', str(rms_mix_ratio),
            '--protect', str(protect),
        ]
        arguments = [argument for argument in arguments if argument]  # Removes all "None" objects in the list.
        env = select_hardware(gpu_id)
        subprocess.run([PYTHON_EXECUTABLE, INFERENCE_SCRIPT_PATH, *arguments], env=env)


    def get_index_path(character):
        character_dir = get_model_path(ARCHITECTURE_NAME, character)
        index_path = None
        try:
            index_path = get_single_file_with_extension(character_dir, INDEX_FILE_EXTENSION)
        except Exception as e:
            print('No ' + INDEX_FILE_EXTENSION + ' file was found in ' + character_dir, flush=True)
        return index_path


    def copy_output(output_filename_sans_extension, session_id):
        array_output, sr_output = read_audio(os.path.join(OUTPUT_COPY_FOLDER,
                                                          output_filename_sans_extension + TEMP_FILE_EXTENSION))
        cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array_output, sr_output)


    def get_temp_files():
        return [os.path.join(INPUT_COPY_FOLDER, file) for file in os.listdir(INPUT_COPY_FOLDER)] + \
               [os.path.join(OUTPUT_COPY_FOLDER, file) for file in os.listdir(OUTPUT_COPY_FOLDER)]


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py', description='A webservice interface for voice conversion with RVC')
    parser.add_argument('--cache_implementation', default='file', choices=cache_implementation_map.keys(), help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = select_cache_implementation(args.cache_implementation)
    register_methods(cache)
    app.run(host='0.0.0.0', port=6578)
