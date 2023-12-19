from django import forms


class UploadImgForm(forms.Form):
    image_file = forms.ImageField(widget=forms.FileInput(attrs={'id': 'image-input'}))