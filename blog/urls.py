from django.urls import path
from . import views
from . views import PostCreateView, PostUpdateView, PostDeleteView, download_combined_pdf
urlpatterns =[
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('Try/', views.Try, name='Try'),
    path('MainPage/', views.Try, name='MainPage'),
    path('detail/<int:post_id>/', views.post_detail, name='detail'),
    path('new_post/', PostCreateView.as_view(), name='new_post'),
    path('detail/<slug:pk>/update/', PostUpdateView.as_view(), name='post-update'),
    path('detail/<slug:pk>/delete/', PostDeleteView.as_view(), name='post-delete'),
    #path('process_upload', views.process_upload, name='process_upload'),
    #path('download_combined_pdf', views.download_combined_pdf, name='download_combined_pdf'),
    path('download_pdf/<int:post_id>/', download_combined_pdf, name='download_pdf'),
]

