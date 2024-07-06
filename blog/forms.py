from django import forms
from .models import Comment, Post
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Div
class NewComment(forms.ModelForm) :
    class Meta:
        model = Comment
        fields = ('name','email','body')
        
#add one mode field to take the compliance file from user
class PostCreateForm(forms.ModelForm) :
    title = forms.CharField(label='عنوان التقرير', max_length=50)
    content = forms.CharField(label=' وصف التقرير ', widget=forms.Textarea )
    uploaded_report = forms.FileField(label='ارفاق الملف', required=True)
   
    
    class Meta:
        model = Post
        fields = ['title','content','uploaded_report']


class PostUpdateForm(forms.ModelForm):
    title = forms.CharField(label='عنوان التقرير', max_length=50)
    content = forms.CharField(label=' وصف التقرير ', widget=forms.Textarea)
    
    class Meta:
        model = Post
        fields = ['title', 'content']