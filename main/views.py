from django.shortcuts import render, redirect
from .models import AudioFile
import whisper
import chromadb
import os
from pydub import AudioSegment
from .embedfunction import Embedder

threshold = 0.3 # for cosine distance
# Use larger models for better results (medium, turbo and large)
model = whisper.load_model("base")

client = chromadb.PersistentClient(path="chromadb_transcripts")
embedder = Embedder("bge-m3")

# cosine returns cosine distance
collection = client.get_or_create_collection(name="transcripts", embedding_function=embedder, metadata={"hnsw:space": "cosine"}) 

def upload_audio(request):
    if not request.user.is_superuser:
        return redirect('search')
    
    if request.method == 'POST' and request.FILES.getlist('audio_files'):
        # Get the list of uploaded audio files
        for audio in request.FILES.getlist('audio_files'):
            # Save the uploaded audio file to the database
            audio_file = AudioFile.objects.create(audio=audio)
            audio_path = audio_file.audio.path  # Path where the audio is saved locally
            
            # Load the audio file using pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Check if the audio duration is longer than 30 seconds
            if len(audio) > 30 * 1000:  # Duration in milliseconds
                segments = []
                for i in range(0, len(audio), 30 * 1000):  # Split into 30-second chunks
                    segment = audio[i:i + 30 * 1000]
                    segments.append(segment)
            else:
                segments = [audio]  # If <=30s, use the whole file
            
            # Process each segment
            for segment in segments:
                # Save the segment temporarily
                segment_path = f"{os.path.splitext(audio_path)[0]}_segment_{segments.index(segment)}.mp3"
                segment.export(segment_path, format="mp3")
                
                # Transcribe the segment
                result = model.transcribe(segment_path)
                transcript = result.get('text', '').strip()
                
                # Clean up the temporary segment file
                os.remove(segment_path)
                # Skip empty transcriptions
                if not transcript:
                    continue

                # Generate metadata for Chroma
                metadata = {"audio_id": audio_file.id, "segment_index": segments.index(segment)}
                # Add the transcript and embedding into the Chroma collection
                collection.add(
                    documents=[transcript],  # Use the transcribed text as a document
                    ids=[str(collection.count() + 1)],  # Auto-incrementing ID for Chroma
                    metadatas=[metadata]  # Include metadata with audio_id and segment_index
                )
                

        return render(request, 'upload.html')  # Redirect to the list of uploaded audios

    return render(request, 'upload.html')

def search(request):
    if request.method == "GET" and request.GET.get("query"):
        audios = []
        # Retrieve and process the query
        message = request.GET.get("query").strip()
        transcripts = collection.query(
            query_embeddings=[embedder.embed_query(message)],
            n_results=20,
        )
        if not transcripts:
            return render(request, 'search.html', {'audios': audios})
        
        print(transcripts)
        filtered_metadata = [
            metadata for distance, metadata in zip(transcripts['distances'][0], transcripts['metadatas'][0]) 
            if (distance < threshold)
        ]


        # Deduplicate transcript IDs before querying the database
        unique_audio_ids = set(
            metadata["audio_id"] for metadata in filtered_metadata
        )
        
        # Fetch all related audio files in a single query
        audios = AudioFile.objects.filter(id__in=list(unique_audio_ids)[:10]) # return max 10 audios
        return render(request, 'search.html', {'audios': audios})

    return render(request, 'search.html')
