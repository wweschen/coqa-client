from django.db import models

# Create your models here.
#
class CoqaStory(models.Model):
    story = models.CharField(max_length=8000)

class QA(models.Model):
    turn = models.IntegerField()
    question = models.CharField(max_length=200)
    answer =models.CharField(max_length=100,blank=True)
    story = models.ForeignKey(CoqaStory, on_delete=models.CASCADE)





