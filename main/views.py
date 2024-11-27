from django.shortcuts import render, redirect
from .models import AudioFile
import whisper
import chromadb
from .embedfunction import Embedder

model = whisper.load_model("base")

client = chromadb.PersistentClient(path="chromadb_transcripts")
collection = client.get_or_create_collection(name="transcripts", embedding_function=Embedder("bge-m3"), metadata={"hnsw:space": "cosine"})

def upload_audio(request):
    if request.method == 'POST' and request.FILES.getlist('audio_files'):
        # Get the list of uploaded audio files
        for audio in request.FILES.getlist('audio_files'):
            audio_file = AudioFile.objects.create(audio=audio)
            audio_path = audio_file.audio.path  # Assuming audio is saved locally
            result = model.transcribe(audio_path)
            transcript = result['text']
            print(transcript)
            # Generate metadata for Chroma
            metadata = {"audio_id": audio_file.id}  # Associate audio file with the metadata

            # Embed and add the transcript into Chroma collection
            collection.add(
                documents=[transcript],  # Use the transcribed text as a document
                ids=[str(audio_file.id)],  # Use audio file ID as the unique ID for the document
                metadatas=[metadata]  # Include metadata with audio_id
            )

        return redirect('audio_list')  # Redirect to the list of uploaded audios
    
    return render(request, 'upload.html')

def audio_list(request):
    # Get all the uploaded audio files
    audios = AudioFile.objects.all()
    return render(request, 'search.html', {'audios': audios})
