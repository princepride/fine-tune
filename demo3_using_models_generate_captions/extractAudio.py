import ffmpeg

def extractAudio(input_file, output_file):
    stream = ffmpeg.input(input_file)
    audio = stream.audio
    audio = ffmpeg.output(audio, output_file)
    ffmpeg.run(audio)

extractAudio("./data/bda-030.mp4","./data/bda-030.mp3")
