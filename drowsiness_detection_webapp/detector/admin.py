from django.contrib import admin

from .models import DrowsinessAnalysis


@admin.register(DrowsinessAnalysis)
class DrowsinessAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "source", "state", "is_drowsy", "confidence", "ear", "mar", "head_pitch")
    list_filter = ("source", "state", "is_drowsy")
