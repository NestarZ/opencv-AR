#importations
from direct.showbase.ShowBase import ShowBase
from pandac.PandaModules import *
import direct.directbase.DirectStart
import math
from direct.task import Task

# montrer le graphe de scene et le taux de rafraichissement
loadPrcFileData("", "want-directtools #t")
loadPrcFileData("", "want-tk #t")
loadPrcFileData("", "show-frame-rate-meter #t") # let me see the frames per second


# Mettre le cube dans la scene
cube = loader.loadModel('cube')
cube.setPos(0,0,0)
cube.reparentTo(render)
# Positionner la camera
camera.setPos(0.0, -60, 0.0)
camera.lookAt(0.0, 0.0, -6.0)
base.disableMouse()

def animer(task):
	
	print(task.time)
	cube.setHpr(task.time*200, 0, 0)
	#cube.setPos(0, task.time, 0)
	return Task.cont
taskMgr.add(animer, "animer")


# lancer le graphe de scene
run()
