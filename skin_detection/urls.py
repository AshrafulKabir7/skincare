"""
URL configuration for skin_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from detection import views  # Import the home view
urlpatterns = [
    path('admin/', admin.site.urls),
    path('detection/', include('detection.urls')),  # Include the detection app's URLs
    path('', views.home, name='home'),  # Root URL displays the home view
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
