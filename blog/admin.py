from django.contrib import admin
from .models import Post, Comment

#admin.site.register(Post)
admin.site.register(Comment)

class PostAdmin(admin.ModelAdmin):
    list_display = ['title', 'content', 'uploaded_report_display']  # Add 'uploaded_report_display' to the list display

    def uploaded_report_display(self, obj):
        if obj.uploaded_report:
            return obj.uploaded_report.url  # Display the file URL
        else:
            return "No file attached"

admin.site.register(Post, PostAdmin)