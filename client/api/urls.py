from rest_framework import routers
from .api import StoryViewSet,QAViewSet

router = routers.DefaultRouter()
router.register('api/story', StoryViewSet, 'story')
router.register('api/qa', QAViewSet, 'qa')

urlpatterns = router.urls