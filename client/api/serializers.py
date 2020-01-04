from rest_framework import serializers
from api.models import CoqaStory, QA

class QASerializer(serializers.ModelSerializer):
  class Meta:
    model = QA
    fields = '__all__'

class StorySerializer(serializers.ModelSerializer):
  qa_set = QASerializer(read_only=True, many=True)
  class Meta:
    model = CoqaStory
    depth =1
    fields = (
            'id',
            'story',
            'qa_set',
        )

