from django.db import models

class AudioFile(models.Model):
    audio = models.FileField(upload_to='audio/')  # Store files in the 'audio' directory
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Audio uploaded at {self.uploaded_at}"
