from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import os
import logging
from urllib.parse import urlparse
from deepgram import Deepgram
from moviepy.editor import VideoFileClip
from botocore.exceptions import ClientError
import ffmpeg
from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI()

# AWS S3 configuration
AWS_ACCESS_KEY = os.getenv('VITE_PUBLIC_S3_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('VITE_PUBLIC_S3_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('VITE_PUBLIC_S3_REGION')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
print(DEEPGRAM_API_KEY, AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY)
s3 = boto3.client('s3',
                  aws_access_key_id=AWS_ACCESS_KEY,
                  aws_secret_access_key=AWS_SECRET_KEY,
                  region_name=AWS_REGION)


class BodyInput(BaseModel):
    s3_url: str


@app.post("/transcribe-s3-video/")
async def transcribe_s3_video(data: BodyInput):
    try:
        # Parse the S3 URL
        s3_url = data.s3_url
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc.split('.')[0]
        key = parsed_url.path.lstrip('/')

        # Set up file paths
        filename = os.path.basename(key)
        base_filename, _ = os.path.splitext(filename)
        video_path = f'./{filename}'
        audio_path = f'./{base_filename}.wav'

        # Download video from S3
        logger.info(f"Downloading video from S3: {bucket_name}/{key}")
        s3.download_file(bucket_name, key, video_path)
        logger.info(f"Video downloaded successfully: {video_path}")

        # Extract audio from the video file
        logger.info("Extracting audio from video")
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path)
            audio_clip.close()
            video_clip.close()
            logger.info(f"Audio extracted successfully using MoviePy: {audio_path}")
        except Exception as e:
            logger.warning(f"MoviePy failed to extract audio, trying ffmpeg: {str(e)}")
            try:
                ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').overwrite_output().run()
            except ffmpeg.Error as e:
                logger.error(f"ffmpeg failed to extract audio: {e.stderr.decode()}")
                raise

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.error(f"Audio file is missing or empty: {audio_path}")
            raise Exception(f"Audio file is missing or empty: {audio_path}")

        # Transcribe audio using Deepgram
        transcription = await transcribe_with_deepgram(audio_path)

        # Clean up files
        os.remove(video_path)
        os.remove(audio_path)

        logger.info("Transcription completed successfully")
        return {"recognized_speech": transcription.strip()}

    except ClientError as e:
        logger.error(f"Error accessing S3: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Error accessing S3: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


async def transcribe_with_deepgram(audio_path: str) -> str:
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    transcription = ""

    try:
        with open(audio_path, 'rb') as audio_file:
            source = {'buffer': audio_file, 'mimetype': 'audio/wav'}
            logger.info("Sending audio to Deepgram...")
            response = await dg_client.transcription.prerecorded(
                source,
                {
                    'punctuate': True,
                    'diarize': True,
                    'utterances': True,
                    'profanity_filter': False,
                    'filler_words': True
                }
            )

            if response.get('results', {}).get('utterances'):
                transcription = ' '.join(
                    utterance['transcript'] for utterance in response['results']['utterances']
                )
                logger.info("Transcription including filler words:")
                logger.info(transcription)
            else:
                logger.warning("No utterances detected in the response.")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise Exception(f"Transcription failed: {e}")

    return transcription


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8002)