

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Get Started with TensorFlow Graphics
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/graphics"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/_index.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/g3doc/_index.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>
If Tensorflow Graphics and Trimesh are not installed on your system the following cell can install these packages for you.
"""

!pip install tensorflow-graphics
!pip install trimesh

"""Now that Tensorflow Graphics and Trimesh are installed, let's import everything needed."""

import numpy as np
import tensorflow as tf
import trimesh

import tensorflow_graphics.geometry.transformation as tfg_transformation
from tensorflow_graphics.notebooks import threejs_visualization

"""We can now load a mesh and rotate it using TensorFlow Graphics."""

# Download the mesh.
# Courtesy of Keenan Crane www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/.
!wget -N https://cdn.discordapp.com/attachments/716742625515798610/801130786442313788/bs_smile.obj
# Load the mesh.
mesh = trimesh.load("bs_smile.obj")
mesh = {"vertices": mesh.vertices, "faces": mesh.faces}
# Visualize the original mesh.
_ = threejs_visualization.triangular_mesh_renderer(mesh, width=800, height=800)
# Set the axis and angle parameters.
axis = np.array((0., 1., 0.))  # y axis.
angle = np.array((np.pi / 4.,))  # 45 degree angle.
# Rotate the mesh.
mesh["vertices"] = tfg_transformation.axis_angle.rotate(mesh["vertices"], axis,
                                                        angle).numpy()
# Visualize the rotated mesh.
_ = threejs_visualization.triangular_mesh_renderer(mesh, width=800, height=800)

#Zastosowanie
1) WildEye - projekt zajmujący się ochrną dzikich zwierząt. Cel projektu to walka z nielegalnymi polowaniami oraz handlem zwierzetami. Tensorflow jest wykorzystywane do przeglądania obrazu z wielu kamer jednocześnie,zbierać dane i wysyłać alarm o zagrożeniu.

2) SIGHT - Okulary wspomagające umiejętności poznawcze ludzi niewidomych, aparat i kamery analizują obraz wokół,a po jego przetworzeniu podają opis przy pomocy głośników.

3) Farmaid - Robot-lekarz rośłin, który przy użyciu kamer oraz Tenserflow zbiera obraz i analizuje czy dana roślina jest chora
Robot autonomicznie jeżdżący wśród roślin w szklarni w celu znajdywania i identyfikacji chorób roślinnych. Dzięki Tensorflow oraz kamerom może analizować obraz, aby zidentyfikować chorobę oraz może przemieszczać się nie uszkadzając niczego.  
