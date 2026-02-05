import base64
import tempfile

def save_base64_audio(base64_string):
    audio_bytes = base64.b64decode(base64_string)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    temp_file.write(audio_bytes)
    temp_file.close()

    return temp_file.name
