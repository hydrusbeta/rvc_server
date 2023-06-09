from hay_say_common import ROOT_DIR, PREPROCESSED_DIR, OUTPUT_DIR, CACHE_EXTENSION, get_model_path, clean_up, \
    construct_full_error_message, read_audio, save_audio_to_cache, create_link

from flask import Flask, request
import jsonschema

import os.path
import traceback
import json
import subprocess
import base64
import shutil


ARCHITECTURE_NAME = 'rvc'
ARCHITECTURE_ROOT = os.path.join(ROOT_DIR, ARCHITECTURE_NAME)
INPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'input')
OUTPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'output')
WEIGHTS_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'weights')
INDICES_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'logs')

PYTHON_EXECUTABLE = os.path.join(ROOT_DIR, '.venvs', ARCHITECTURE_NAME, 'bin', 'python')
INFERENCE_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, 'command_line_interface.py')

INDEX_FILE_EXTENSION = '.index'
WEIGHTS_FILE_EXTENSION = '.pth'

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate() -> (str, int):
    code = 200
    message = ""
    try:
        character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius, rms_mix_ratio, \
            protect, output_filename_sans_extension = parse_inputs()
        link_model_path(character)
        copy_input_audio(input_filename_sans_extension)
        execute_program(character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius,
                        rms_mix_ratio, protect, output_filename_sans_extension)
        copy_output(output_filename_sans_extension)
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


enum_mapper = {'crepe': 'crepe', 'harvest': 'harvest', 'parselmouth': 'pm'}


def parse_inputs():
    schema = {
        'type': 'object',
        'properties': {
            'Inputs': {
                'type': 'object',
                'properties': {
                    'User Audio': {'type': 'string'}
                }
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
                }
            },
            'Output File': {'type': 'string'}
        }
    }

    jsonschema.validate(instance=request.json, schema=schema)

    input_filename_sans_extension = request.json['Inputs']['User Audio']
    character = request.json['Options']['Character']
    pitch_shift = request.json['Options']['Pitch Shift']
    f0_method = request.json['Options']['f0 Extraction Method']
    index_ratio = request.json['Options']['Index Ratio']
    filter_radius = request.json['Options']['Filter Radius']
    rms_mix_ratio = request.json['Options']['Voice Envelope Mix Ratio']
    protect = request.json['Options']['Voiceless Consonants Protection Ratio']
    output_filename_sans_extension = request.json['Output File']

    return character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius, \
        rms_mix_ratio, protect, output_filename_sans_extension


class BadInputException(Exception):
    pass


def link_model_path(character):
    """Create a symbolic link to the model file in the location where RVC expects to find it."""
    symlink_file = os.path.join(WEIGHTS_FOLDER, character + WEIGHTS_FILE_EXTENSION)
    character_dir = get_model_path(ARCHITECTURE_NAME, character)
    weight_file = os.path.join(character_dir, character + WEIGHTS_FILE_EXTENSION)
    create_link(weight_file, symlink_file)


def copy_input_audio(input_filename_sans_extension):
    source = os.path.join(PREPROCESSED_DIR, input_filename_sans_extension + CACHE_EXTENSION)
    target = os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + CACHE_EXTENSION)
    try:
        shutil.copyfile(source, target)
    except Exception as e:
        raise Exception("Unable to copy file from Hay Say's audio cache to rvc's raw directory.") from e


def execute_program(character, input_filename_sans_extension, pitch_shift, f0_method, index_ratio, filter_radius,
                    rms_mix_ratio, protect, output_filename_sans_extension):
    arguments = [
        '--voice', character + WEIGHTS_FILE_EXTENSION,
        '--sid', str(0),
        '--input_filepath', os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + CACHE_EXTENSION),
        '--transpose', str(pitch_shift),
        '--f0_method', f0_method,
        '--index_filepath', get_index_path(character),
        '--index_ratio', str(index_ratio),
        '--filter_radius', str(filter_radius),
        '--resample_rate', str(0),
        '--rms_mix_ratio', str(rms_mix_ratio),
        '--protect', str(protect),
        '--output_filepath', os.path.join(OUTPUT_COPY_FOLDER, output_filename_sans_extension + CACHE_EXTENSION)
    ]
    subprocess.run([PYTHON_EXECUTABLE, INFERENCE_SCRIPT_PATH, *arguments])


def get_index_path(character):
    character_dir = get_model_path(ARCHITECTURE_NAME, character)
    return os.path.join(character_dir, character + INDEX_FILE_EXTENSION)


def copy_output(output_filename_sans_extension):
    array_output, sr_output = read_audio(os.path.join(OUTPUT_COPY_FOLDER,
                                                      output_filename_sans_extension + CACHE_EXTENSION))
    save_audio_to_cache(OUTPUT_DIR, output_filename_sans_extension, array_output, sr_output)


def get_temp_files():
    return [os.path.join(INPUT_COPY_FOLDER, file) for file in os.listdir(INPUT_COPY_FOLDER)] + \
           [os.path.join(OUTPUT_COPY_FOLDER, file) for file in os.listdir(OUTPUT_COPY_FOLDER)]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6578)
