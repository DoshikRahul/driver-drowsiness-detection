"""
URL patterns for the detector app
"""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_image, name="upload_image"),
    path("live/", views.live_monitor, name="live_monitor"),
    path("result/<int:analysis_id>/", views.analysis_result, name="analysis_result"),
    path("history/", views.history, name="history"),
    path("download/<int:analysis_id>/", views.download_report, name="download_report"),
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/analyze-frame/", views.api_analyze_frame, name="api_analyze_frame"),
    path("model-info/", views.model_info, name="model_info"),
]
