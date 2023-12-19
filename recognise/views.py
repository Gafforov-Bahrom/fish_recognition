from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .forms import UploadImgForm
from .models import RecognisionModel
import logging
import cv2
from . import utils
from .utils import *


def handle_uploaded_file(filename, f):
    with open(filename, "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                user:User
                user.first_name = request.POST.get("first_name", user.first_name)
                user.last_name = request.POST.get("last_name", user.last_name)
                user.save()
                login(request, user)
                return redirect('home') 
        else:
            logging.error(form.errors)
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect("home")

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            print("redirecting")
            return redirect('home')  # replace 'home' with the name of your homepage url
        return render(request, 'login.html', context={"error": "Invalid username or password"})
    return render(request, 'login.html')


def profile_view(request):
    if request.method == 'POST':
        user = request.user
        user.first_name = request.POST.get("first_name", user.first_name)
        user.last_name = request.POST.get("last_name", user.last_name)
        user.save()
        return redirect("home")
    return render(request, 'profile.html')

def home(request):
    form = UploadImgForm()
    if request.method == "POST":
        form = UploadImgForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file("media/temp.png", request.FILES["image_file"])
            recognision_model = RecognisionModel.objects.filter(is_active=True).first()
            img = cv2.imread("media/temp.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = utils.get_transforms(False)(image=img, bboxes=[[10, 10, 1, 1, "label"]])
            img = transformed['image']
            img = img.div(255)
            species = recognision_model.recognise(img)
            return render(request, 'home.html', context={"form": UploadImgForm(request.POST, request.FILES), 
                                                         "img_file": "temp_res.png", "height": "360px", "width": "640px", "img_request_file": "temp.png", "species": species,
                                                         "description": descriptions[species]})
        else:
            logging.error(form.errors)
    return render(request, 'home.html', context={"form": form})
