from typing import Optional
from django.shortcuts import render , get_object_or_404, redirect
from .models import Post, Comment

from django.core.files.base import ContentFile

from .forms import NewComment, PostCreateForm, PostUpdateForm
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.views.generic import CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django .contrib import messages
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from io import BytesIO

from django.http import HttpResponse
from .prepocessing import DataPreprocessor
import os
import pandas as pd
#from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import pickle
import io
#creating views

def Try(request) :
    return render(request, 'blog/Try.html', context={'title':'Try'})

def MainPage(request) :
    return render(request, 'blog/TrMainPagey.html', context={'title':'MainPage'})

def home(request) :
    posts = Post.objects.all()
    paginator = Paginator(posts, 5)
    page = request.GET.get('page')
    try:
        posts = paginator.page(page)
    except PageNotAnInteger:
        posts = paginator.page(1)
    except EmptyPage:
        posts = paginator.page(paginator.num_pages)

    context ={ 
    'title': 'الصفحة الرئيسية',
    'posts' : posts,
    'page' : page,
    }
    #blog/index.html refers to the file at the temblates folder 
    return render(request, 'blog/index.html', context)

def about(request) :
    return render(request, 'blog/about.html', context={'title':'من أنا'})

def post_detail (request, post_id):
    post=  get_object_or_404(Post,pk=post_id)
    comments = post.comments.filter(active=True) #show the comment if it is activated
    #التأكد من صحة البيانات قبل الحفظ
    if request.method == 'POST':
        comment_form = NewComment(data=request.POST)
        if comment_form.is_valid():
            new_comment = comment_form.save(commit=False)
            new_comment.post = post
            new_comment.save()
            comment_form = NewComment () #empty the form
    else:
        comment_form = NewComment()

    context={
    'title': post,
    'post': post,
    'comments' : comments,
    'comment_form': comment_form,
    }

    return render(request, 'blog/detail.html', context)

class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    template_name = 'blog/new_post.html'
    form_class = PostCreateForm

    def form_valid(self, form):
        print(self.request.user)
        uploaded_report = self.request.FILES.get('uploaded_report', None)
        if uploaded_report:
            # Check if the file is an Excel file
            if uploaded_report.name.endswith('.xlsx') or uploaded_report.name.endswith('.xls'):
                if uploaded_report.size > 0:
                    fs = FileSystemStorage()
                    filename = fs.save(uploaded_report.name, uploaded_report)

                    # Get the full file path
                    file_path = os.path.join(fs.location, filename)

                    # Initialize the preprocessor with the uploaded file
                    preprocessor = DataPreprocessor(file_path)

                    # Perform preprocessing
                    preprocessed_data = preprocessor.preprocess()

                    # Store file path in session
                    self.request.session['uploaded_file_path'] = file_path
                    
                    # Generate PDF
                    pdf_data = DataPreprocessor.mergepdf(preprocessed_data)  # Assuming a function generate_pdf is defined

                    # Create a variable for the ContentFile
                    #pdf_content = pdf_data.getvalue()  # Get the content of the PDF file

                    # Create a ContentFile from the PDF data
                    pdf_content = ContentFile(pdf_data.getvalue(), name='generated_pdf.pdf')  # New Line


                    # Save the generated PDF to the Post model
                    # form.instance.generated_pdf.save('generated_pdf.pdf', ContentFile(pdf_content))

                    # Set the author and process the form
                    form.instance.author = self.request.user
                    form.instance.preprocessed_data = preprocessed_data # Assuming the Post model has a field for this

                    # Save the generated PDF to the Post model
                    form.instance.generated_pdf.save(pdf_content.name, pdf_content)  # New Line

                    return super().form_valid(form)
                else:
                    # File is empty, display warning message
                    messages.warning(self.request, 'الملف المرفق فارغ!')
                    return super().form_invalid(form)
            else:
                # File is not an Excel file, display warning message
                messages.warning(self.request, 'يرجى إرفاق ملف Excel فقط!')
                return super().form_invalid(form)
        else:
            # File not uploaded, display warning message
            messages.warning(self.request, 'يرجى إرفاق تقرير الالتزام قبل التأكيد!')
            return super().form_invalid(form)
        

def download_combined_pdf(request, post_id):
    # Retrieve the Post instance by ID
    post = get_object_or_404(Post, id=post_id)
    
    # Get the PDF file associated with the Post instance
    if post.generated_pdf:
        # Get the PDF file content
        pdf_file = post.generated_pdf

        response = HttpResponse(pdf_file.read(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{pdf_file.name}"'
        return response
    
    else:
        messages.error(request, 'No PDF file found for this post.')
        return redirect('upload')  # Redirect to a relevant page

class PostUpdateView(UserPassesTestMixin,LoginRequiredMixin,UpdateView):
    model = Post 
    template_name = 'blog/post_update.html'
    form_class = PostUpdateForm

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
    
    def test_func(self):
       post = self.get_object()
       if self.request.user == post.author:
           return True
       else:
           return False
       
class PostDeleteView(UserPassesTestMixin, LoginRequiredMixin, DeleteView):
    model = Post
    success_url = '/'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False
    


""" file_path=''
def process_upload(request):
    if request.method == 'POST' and request.FILES['xlsx_file']:
        xlsx_file = request.FILES['xlsx_file']
        fs = FileSystemStorage()
        filename = fs.save(xlsx_file.name, xlsx_file)

        uploaded_file_url = fs.url(filename)
        
        # Load the XLSX file into a DataFrame
        file_path = os.path.join(fs.location, filename)
        #file_path1 = os.path.join(fs.location, filename)

        # Initialize the preprocessor with the uploaded file
        preprocessor = DataPreprocessor(file_path)

        # Perform preprocessing
        list = preprocessor.preprocess()

        return render(request, 'blog/user.html')
    return redirect('upload')
 """

""" @login_required
def download_combined_pdf(request):

    preprocessor1 = DataPreprocessor(file_path)
    pdf=preprocessor1.mergepdf() 
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename=combined.pdf'
    #response['Content-Length'] = len(pdf_data)
    response.write(pdf.getvalue())
    print("hello")
    return response """