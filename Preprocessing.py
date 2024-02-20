import os
import open3d as o3d
from PIL import Image
import numpy as np
import random


def process_stl_files(input_folder, output_folder):
        print("Data Creation And Agumentation Progress is started.....")
        os.makedirs(output_folder, exist_ok=True)

        rotation_angles_top = range(0, 361, 10)  
        rotation_angles_front = range(0, 361, 10)  
        scaling_factors = [2.0, 1.5, 1.0, 0.5, 0.1]  
        translation_values = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),  
            (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2), 
            (3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0), (0, 0, 3), (0, 0, -3),  
        ]

        random.seed(42)

        def create_random_color(mesh):
            color = np.random.rand(3)
            mesh.paint_uniform_color(color)
            return mesh
        for filename in os.listdir(input_folder):
            if filename.endswith(".stl"):
                stl_file = os.path.join(input_folder, filename)

                # Read the STL file
                mesh = o3d.io.read_triangle_mesh(stl_file)

                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_non_manifold_edges()
                mesh.compute_vertex_normals()

                mesh.normalize_normals()

                for scaling_factor in scaling_factors: 
                    mesh_rotated_scaled=mesh.scale(scaling_factor, center=mesh.get_center())
                    for translation in translation_values:  
                        translated_mesh=mesh_rotated_scaled.translate(translation)

                        # Apply top-to-top augmentation
                        for rotation_angle_top in rotation_angles_top:
                            try:
                                mesh = create_random_color(translated_mesh)
                                vis = o3d.visualization.Visualizer()
                                vis.create_window()
                                vis.add_geometry(mesh)
                                mesh_rotated_scaled = mesh.rotate(
                                    o3d.geometry.get_rotation_matrix_from_xyz((np.radians(rotation_angle_top), 0, 0)))
                                
                                vis.add_geometry(mesh_rotated_scaled)
                                def capture_screen(vis):
                                    image = vis.capture_screen_float_buffer(do_render=True)
                                    return np.asarray(image)
                                image = capture_screen(vis)
                                vis.destroy_window()
                                image_pil = Image.fromarray((image * 255).astype(np.uint8))
                                random_number = random.randint(1000, 9999)
                                augmented_filename = f"img_{os.path.splitext(filename)[0]}_top_{random_number}.png"
                                output_file = os.path.join(output_folder, augmented_filename)
                                image_pil.save(output_file)

                            except Exception as e:
                                print(f"An error occurred during rendering: {e}")

                        # Apply front-to-front augmentation
                        for rotation_angle_front in rotation_angles_front:
                            try:
                                mesh = create_random_color(translated_mesh)
                                vis = o3d.visualization.Visualizer()
                                vis.create_window()
                                vis.add_geometry(mesh)
                                mesh_rotated_scaled = mesh.rotate(
                                    o3d.geometry.get_rotation_matrix_from_xyz((0, np.radians(rotation_angle_front), 0)))
                                
                                vis.add_geometry(mesh_rotated_scaled)
                                def capture_screen(vis):
                                    image = vis.capture_screen_float_buffer(do_render=True)
                                    return np.asarray(image)
                                image = capture_screen(vis)
                                vis.destroy_window()
                                image_pil = Image.fromarray((image * 255).astype(np.uint8))
                                random_number = random.randint(1000, 9999)
                                augmented_filename = f"img_{os.path.splitext(filename)[0]}_front_{random_number}.png"
                                output_file = os.path.join(output_folder, augmented_filename)
                                image_pil.save(output_file)

                            except Exception as e:
                                print(f"An error occurred during rendering: {e}")

                print("all Files are generated")