import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import trimesh as tm
import click
import os
from time import time

BASE_COLOR = (0.5,0.5,0.7)
SELECTED_COLOR = (0.9,0.0,0.0)
CURSOR_COLOR = (0.4,0.8,0.4)
SCALE_FACTOR = 1.2

COMMANDS = dict(
    mask_selection=('space', 'Space', 'Mark the area under the cursor as masked'),
    unmask_selection=('Control_L', 'Left control', 'Unmask the area under the cursor if masked'),
    increase_area_size =('plus', 'Numpad Plus', 'Increase the selection area size'),
    decrease_area_size =('minus', 'Numpad Minus', 'Decrease the selection area size'),
)

# Utility fonctions

def verts_mask_to_npy_mask_one_by_one(all_verts, mask_verts):
    vids = []
    for v in mask_verts:
        distances = np.linalg.norm(all_verts - v[None, :], axis=-1)
        vids.append(distances.argmin())
    mask_index = np.array(vids)
    mask_bool = np.zeros(all_verts.shape[0], dtype=bool)
    mask_bool[mask_index] = True
    return mask_index, mask_bool

def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    _, inter, _ = np.intersect1d(arr1_view, arr2_view, return_indices=True)
    return inter

def verts_mask_to_npy_mask(full_verts, mask_verts, precision=None):
    if precision is not None:
        inter_index = multidim_intersect(np.around(full_verts, precision),
                                         np.around(mask_verts, precision))
    else:    
        inter_index = multidim_intersect(full_verts, mask_verts)
    
    inter_bool = np.zeros(full_verts.shape[0], dtype=bool)
    inter_bool[inter_index] = True

    return inter_index, inter_bool

def tm_mask_to_npy_mask(all_mesh, mask_mesh, precision=None, target_ccomps=None):
    ccomps = tm.graph.connected_component_labels(all_mesh.edges)
    nb_ccomps = ccomps.max() + 1
    if target_ccomps is None:
        target_ccomps = range(nb_ccomps)
    nV = all_mesh.vertices.shape[0]
    global_mask_bool = np.zeros(nV, dtype=bool)
    for c in range(nb_ccomps):
        if c in target_ccomps:
            
            ccomp_mask = np.zeros(all_mesh.vertices.shape[0], dtype=bool)
            ccomp_mask[ccomps == c] = True
             
            _, ccomp_inter_bool = verts_mask_to_npy_mask(0.01 * all_mesh.vertices[ccomp_mask, :], 0.01 * mask_mesh.vertices, precision=precision)
            global_mask_bool[ccomp_mask] = ccomp_inter_bool
    
    global_mask_index = np.where(global_mask_bool)[0]

    return global_mask_index, global_mask_bool

def verts_mask_to_npy_mask_one_by_one(all_verts, mask_verts):
    vids = []
    for v in mask_verts:
        distances = np.linalg.norm(all_verts - v[None, :], axis=-1)
        vids.append(distances.argmin())
    mask_index = np.array(vids)
    mask_bool = np.zeros(all_verts.shape[0], dtype=bool)
    mask_bool[mask_index] = True
    return mask_index, mask_bool

def tm_mask_to_npy_mask_one_by_one(all_mesh, mask_mesh):
    return verts_mask_to_npy_mask_one_by_one(all_mesh.vertices, mask_mesh.vertices)

# Landmarking manager class

class Manager():
    def _do_nothing(*args):
        pass

    def __init__(self, pv_mesh, mesh, mask):
        self.pv_mesh = pv_mesh # pyvista trimeh wrap
        self.mesh = mesh # trimesh mesh
        self.nV = self.mesh.vertices.shape[0]
        self.mask = mask  # index mask to edit
        self.base_colors = np.repeat([BASE_COLOR], self.nV, axis=0)
        self.pv_mesh['colors'] = self.base_colors
        self.scale = self.mesh.scale # helps setting the cursor's size
        self.p = pvqt.BackgroundPlotter()
        self.p.background_color = '#0D1017'
        self.p.iren.picker = 'world'
        self.p.set_icon('mesh_masking.jpg')
        def deactivate_plotter():
            self.p.active=False
            self.p.close()
        self.p.app_window.signal_close.connect(deactivate_plotter)
        self.actor = self.p.add_mesh(self.pv_mesh, scalars='colors', specular=0.5, rgb=True, show_edges=True, edge_color='black', line_width=0.1)

        # Small hack to disable the edge width modification with plus and minus
        self.p.iren._key_press_event_callbacks['plus'] = [Manager._do_nothing]
        self.p.iren._key_press_event_callbacks['minus'] = [Manager._do_nothing]
  
        self.p.key_press_event_signal.connect(self.process_key_press_event) # not implemented yet
        self.p.track_mouse_position() # To track mouse position
        self.pos = (0,0,0) # initial cursor position
        self.cursor_mask = np.zeros_like(self.mask)
        self.cursor_size = 0.033
        self.updated_size = False
        self.cursor = None
        self.cursor_sphere_duration = 0.1
        self.bt = 0

    def run_app(self):
        
        while self.p.active:
            self.pos = self.p.pick_mouse_position()
            if self.updated_size:
                self.cursor = self.p.add_mesh(pv.Sphere(self.cursor_size * self.scale, self.pos), name='cursor', color=CURSOR_COLOR, opacity=0.5, reset_camera=False, smooth_shading=True)
                self.updated_size = False
            else:
                if time() - self.bt > self.cursor_sphere_duration:
                    self.p.remove_actor(self.cursor)
                distances = np.linalg.norm(self.mesh.vertices - np.array(self.pos)[None, :],axis=-1)
                self.cursor_mask = distances < self.scale * self.cursor_size#np.logical_and(distances < self.scale * self.cursor_size, self.get_visible_vertices())
                self.pv_mesh['colors'][:] = BASE_COLOR
                self.pv_mesh['colors'][self.cursor_mask] = CURSOR_COLOR
                self.pv_mesh['colors'][self.mask] = SELECTED_COLOR
                self.p.update_scalars(mesh=self.pv_mesh, scalars=self.pv_mesh['colors'])
            self.p.app.processEvents()

    def process_key_press_event(self, obj, _):
        self.changed = False
        code = obj.GetKeySym()
        if code == COMMANDS['mask_selection'][0]:
            self.mask[self.cursor_mask] = True
        elif code == COMMANDS['unmask_selection'][0]:
            self.mask[self.cursor_mask] = False
        elif code == COMMANDS['increase_area_size'][0]:
            self.cursor_size *= SCALE_FACTOR
            self.updated_size = True
            self.bt = time()
        elif code == COMMANDS['decrease_area_size'][0]:
            self.cursor_size /= SCALE_FACTOR
            self.updated_size = True
            self.bt = time()
        elif code == 'q':
            print('Saving and quitting.')
            return
        else:
            print(code, ' -> no action')


    # def get_visible_vertices(self):
    #     # perfrom the automatic picking
    #     selector = vtk.vtkOpenGLHardwareSelector()
    #     selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
    #     selector.SetRenderer(self.p.renderer)
    #     selector.SetArea(0,0,self.p.window_size[0],self.p.window_size[1])
    #     selection = selector.Select()

    #     extractor = vtk.vtkExtractSelection()
    #     extractor.SetInputData(0, self.pv_mesh)
    #     extractor.SetInputData(1, selection)
    #     extractor.Update()
    #     points = pv.wrap(extractor.GetOutput()).points
    #     _, mask = verts_mask_to_npy_mask_one_by_one(self.pv_mesh.points, points)
    #     return mask


def print_keys():
    print('#' * 100)
    print('#' * 100)
    for action in COMMANDS:
        _, key, descr = COMMANDS[action]
        print('#\t' + key + ' :\t' + descr)
    print('#' * 100)
    print('#' * 100)

@click.command()
@click.argument('mesh_path', type=click.Path(exists=True), required=True)
@click.argument('mask_path', type=click.Path(exists=False), required=True)
def cli(mesh_path, mask_path):

    print_keys()

    mesh = tm.load(mesh_path, process=False)
    pv_mesh = pv.wrap(mesh)

    if os.path.exists(mask_path):
        if mask_path.endswith('.obj'):
            mask_mesh = tm.load(mask_path, process=False)
            _, mask = tm_mask_to_npy_mask_one_by_one(mesh, mask_mesh)
        else:
            mask = np.load(mask_path)
    else:
        mask = np.zeros(mesh.vertices.shape[0], dtype=bool)
    
    manager = Manager(pv_mesh, mesh, mask)
    manager.run_app()

    if mask_path.endswith('.obj'):
        mask_path = mask_path[:-3] + 'npy'
    np.save(mask_path, mask)

if __name__ == '__main__':
    cli()