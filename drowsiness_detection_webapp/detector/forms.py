"""
Forms for the detector app
"""

from django import forms


class ImageUploadForm(forms.Form):
    """Form for uploading a driver face image"""

    image = forms.ImageField(
        label="Upload Driver Image",
        help_text="A clear image of the driver's face (JPG, PNG or WEBP, max 10MB)",
        widget=forms.FileInput(attrs={
            "class": "form-control",
            "accept": "image/*",
            "style": "display: none;",
        }),
    )

    def clean_image(self):
        image = self.cleaned_data.get("image")
        if image:
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Image file too large. Maximum size is 10MB.")
            content_type = getattr(image, "content_type", "") or ""
            if content_type and not content_type.startswith("image/"):
                raise forms.ValidationError("Please upload a valid image file.")
        return image
