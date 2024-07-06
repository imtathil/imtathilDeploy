import os
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.urls import reverse

#tables translelated into the db
# fields are title , content ...
# db is written using pythonand needed to be transformed to sql using commands 
#1. python manage.py makemigrations from python to sql
#2. python manage.py migrate execute the sqql and create the db

def user_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return os.path.join('user', f'{instance.author.id}', filename)
    
class Post( models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    post_date = models.DateTimeField(default=timezone.now) #created first time when post is created
    post_update = models.DateTimeField(auto_now=True) #updated every time the post is alterd
    author = models.ForeignKey(User, on_delete= models.CASCADE) #user is from django tables
    uploaded_report = models.FileField(upload_to=user_directory_path, blank=True, null=True)
    generated_pdf = models.FileField(upload_to='generated_pdfs/', blank=True, null=True)

    def __str__(self): #to return the title at the site administration
        return self.title
    
    def get_absolute_url (self):
         return reverse('detail', args=[self.pk])

    
    class Meta: #for the post to appear from new to old 
        ordering = ('-post_date',)

class Comment (models.Model): #inherits from models class
    name = models.CharField(max_length=30, verbose_name="الإسم")
    email = models.EmailField(verbose_name="البريد الإلكتروني")
    body = models.TextField(verbose_name="التعليق")
    comment_date = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False)
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    def __str__(self):
        return 'علق {} على {}.'.format(self.name, self.post)
    class Meta:
        ordering = ('-comment_date',)




    