from django import forms
from .models import AudioFile

class AudioUploadForm(forms.Form):
    audio_files = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}), required=True)
