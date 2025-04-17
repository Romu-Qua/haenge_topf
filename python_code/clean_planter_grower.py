"""
This is a script that generates a 3D-printable model of a natural growth structure.
It creates a tree-like branching pattern constrained between two cones to ensure printability.
The script requires plant pot base and top files, which should be in the accompanying git repository
or can be obtained from the author at ronny.r_mueller@web.de.

Credit: @Ronny Müller
"""

from typing import List, Tuple, Optional, Union
import bpy
import mathutils
import random
import math
import numpy as np
import bmesh
import uuid  # Add this import at the top


# --- Conical Boundaries ---
TREE_HEIGHT: float = 19.0       # Height of the outer cone
BASE_RADIUS_OUTER: float = 9.4  # Outer cone base radius
BASE_RADIUS_INNER: float = 9.1  # Inner cone base radius
INNER_CONE_HEIGHT: float = 19.5 # Height of the inner cone
MAX_BRANCH_RADIUS: float = 0.25 # Max cylinder radius

# --- Branch Generation Parameters ---
MEAN: float = 0.7               # Mean branch length factor
Z_BRANCH_OFFSET: float = 0.06   # Vertical offset for child branches
STD_DEV: float = 0.1            # Standard deviation for branch length randomness
SINGLE_EXTENDER: float = 1.0    # Multiplier for single branches length
EDGE_LOOP_TOLERANCE: float = 0.1 # Tolerance for edge loop vertex matching
DIRECTION_MODIFIER: float = 0.96 # Modifier for branch length after the end point is constrained

# --- Depth Limits for Branching Phases ---
DEPTH_FIRST_LIMIT: int = 8
DEPTH_SECOND_LIMIT: int = 18

# --- Angle Randomness per Phase ---
# Increase for larger angles but be aware that large angles get cut to make sure the
# print doesn't get any overhangs that require support
ANGLE_FIRST_PHASE: float = 0.4
ANGLE_SECOND_PHASE: float = 0.3

# List to store created objects
created_objects: List[bpy.types.Object] = []


def first_phase_branches_chance() -> int:
    """
    Determines how many child branches to create during the first growth phase.
    
    Returns:
        int: Number of child branches (1 or 2)
    """
    return np.random.choice([1, 2], p=[0.75, 0.25])


def second_phase_branches_chance() -> int:
    """
    Determines how many child branches to create during the second growth phase.
    
    Returns:
        int: Number of child branches (1 or 2)
    """
    return np.random.choice([1, 2], p=[0.9, 0.1])


def last_phase() -> int:
    """
    Determines how many child branches to create during the final growth phase.
    
    Returns:
        int: Number of child branches (1 or 2)
    """
    return np.random.choice([1, 2], p=[0.95, 0.05])


def get_cone_radius(height: float, cone_height: float, base_radius: float) -> float:
    """
    Calculates the radius of a cone at a specific height.
    
    Args:
        height: Current height position
        cone_height: Total height of the cone
        base_radius: Radius at the base of the cone
        
    Returns:
        float: Radius of the cone at the specified height
    """
    return base_radius * (height / cone_height)


def get_cone_surface_normal(point: mathutils.Vector) -> mathutils.Vector:
    """
    Calculates the normal vector to the surface of the cone at the given point.
    
    Args:
        point: 3D point on or near the cone surface
        
    Returns:
        mathutils.Vector: Normal vector to the cone surface
    """
    # Cone parameters
    cone_height = TREE_HEIGHT
    base_radius = BASE_RADIUS_OUTER

    # Calculate radius at this height on the cone
    radius_at_height = get_cone_radius(point.z, cone_height, base_radius)

    # Normal vector is pointing radially outward from the cone's axis
    normal = mathutils.Vector((point.x, point.y, -radius_at_height))  # Vector pointing towards the base
    normal.normalize()

    return normal


def adjust_z_value_if_angle_exceeds_threshold(vector: Union[mathutils.Vector, np.ndarray], 
                                             threshold_deg: float = 60) -> bool:
    """
    Checks if a vector's angle with the Z-axis exceeds a threshold.
    This helps prevent unprintable overhangs.
    
    Args:
        vector: Direction vector to check
        threshold_deg: Maximum allowed angle in degrees
        
    Returns:
        bool: True if angle is acceptable, False if it exceeds the threshold
    """
    # Create a unit vector along the Z-axis (0, 0, 1)
    z_axis = np.array([0, 0, 1])

    # Normalize the vector to ensure it's a unit vector
    vector_norm = vector / np.linalg.norm(vector)

    # Calculate the cosine of the angle between the vector and the Z-axis
    cosine_angle = np.dot(vector_norm, z_axis)

    # Ensure the cosine value is within valid range (-1, 1) due to floating-point precision issues
    cosine_angle = np.clip(cosine_angle, -1, 1)

    # Calculate the angle in radians and convert to degrees
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    # Check if the angle exceeds the threshold
    return angle_deg <= threshold_deg


def get_edge_loop(obj: bpy.types.Object, z_value: float, 
                 tolerance: float = EDGE_LOOP_TOLERANCE) -> List[mathutils.Vector]:
    """
    Finds vertices that form an edge loop at a specific z-coordinate in an object.
    
    Args:
        obj: Blender mesh object to analyze
        z_value: Z-coordinate where to find the edge loop
        tolerance: Distance tolerance for vertex selection
        
    Returns:
        List[mathutils.Vector]: List of vertex coordinates in the edge loop
    """
    if obj.type != 'MESH':
        raise TypeError("Object must be a mesh.")

    obj_matrix_world = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    edge_loop_coords = []
    while len(edge_loop_coords) < 32:
        tolerance += 0.03
        edge_loop_coords = []
        for v in bm.verts:
            world_co = obj_matrix_world @ v.co
            if abs(world_co.z - z_value) < tolerance:
                edge_loop_coords.append(world_co.copy())  # Store vertex coordinates

    bm.free()  # free the bmesh
    
    print(f"Found {len(edge_loop_coords)} vertices in edge loop")
    return edge_loop_coords


def clamp_angle_to_0_78_radians(angle_radians: float) -> float:
    """
    Limits an angle to a maximum of 0.78 radians (approximately 45 degrees)
    while preserving its sign.
    
    Args:
        angle_radians: Angle in radians
        
    Returns:
        float: Clamped angle in radians
    """
    original_sign = 1 if angle_radians >= 0 else -1
    clamped_abs_value = min(abs(angle_radians), 0.78)
    return clamped_abs_value * original_sign


def copy_object_with_suffix(object_name: str) -> None:
    """
    Creates a copy of a Blender object and renames it by appending "copy" to the original name.
    
    Args:
        object_name: Name of the object to copy
    """
    # Get the source object
    source_obj = bpy.data.objects.get(object_name)
    if not source_obj:
        print(f"Source object '{object_name}' not found")
        return
    
    # Create a copy of the object's data
    new_mesh = source_obj.data.copy()  # Copy the mesh data
    new_obj = bpy.data.objects.new(f"{object_name}".replace("_to_copy",""), new_mesh)  # Create a new object with the copied mesh
    
    # Copy the transformation properties
    new_obj.location = source_obj.location.copy()
    new_obj.rotation_euler = source_obj.rotation_euler.copy()
    new_obj.scale = source_obj.scale.copy()
    
    # Link the new object to the current collection
    bpy.context.collection.objects.link(new_obj)
    
    print(f"Created copy of '{object_name}' named '{new_obj.name}'")




def create_branch(start: mathutils.Vector, direction: mathutils.Vector, 
                 depth: int = 0, max_depth: int = 5, 
                 mean: float = MEAN, 
                 Z_branch_offset: float = Z_BRANCH_OFFSET, 
                 std_dev: float = STD_DEV) -> Optional[bpy.types.Object]:
    """
    Creates a branch and recursively generates child branches.
    
    The branch is constrained between inner and outer conical boundaries,
    with directions adjusted to follow the cone surface and avoid unprintable overhangs.
    
    Args:
        start: Starting point of the branch
        direction: Initial direction vector
        depth: Current recursion depth
        max_depth: Maximum recursion depth
        mean: Mean branch length factor
        Z_branch_offset: Vertical offset for child branches
        std_dev: Standard deviation for branch length randomness
        
    Returns:
        Optional[bpy.types.Object]: The created branch object, or None if no branch was created
    """
    # Stop if we've reached the top of the tree
    if start.z >= TREE_HEIGHT:
        return None

    height_fraction = max([start.z / TREE_HEIGHT, 0.3])  # Normalize height (0 at base, 1 at top)

    # Compute valid radial range at this height
    max_radius = get_cone_radius(start.z, TREE_HEIGHT, BASE_RADIUS_OUTER)
    min_radius = get_cone_radius(start.z, INNER_CONE_HEIGHT, BASE_RADIUS_INNER)

    # Constrain the branch within the two cones
    radius_2d = math.sqrt(start.x ** 2 + start.y ** 2)
    if radius_2d > max_radius:
        scale_factor = max_radius / radius_2d
        start.x *= scale_factor
        start.y *= scale_factor
    elif radius_2d < min_radius:
        scale_factor = min_radius / radius_2d
        start.x *= scale_factor
        start.y *= scale_factor

    # Generate branch length with randomness
    sample = np.random.normal(loc=mean, scale=std_dev)
    branch_length = sample
    
    # Calculate branch radius with tapering effect
    branch_radius = MAX_BRANCH_RADIUS * max([1 - height_fraction])
    branch_radius = max(branch_radius, 0.15)  # Prevent vanishingly small branches
    
    # Normalize direction vector and scale by branch length
    direction = direction / math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2) * branch_length
    end = start + direction

    # Ensure end point doesn't exceed maximum height
    if end.z > INNER_CONE_HEIGHT:
        end.z = INNER_CONE_HEIGHT

    # Constrain end point within the conical boundaries
    max_radius = get_cone_radius(end.z, TREE_HEIGHT, BASE_RADIUS_OUTER)
    min_radius = get_cone_radius(end.z, INNER_CONE_HEIGHT, BASE_RADIUS_INNER)

    end_radius_2d = math.sqrt(end.x ** 2 + end.y ** 2)
    if end_radius_2d > max_radius:
        scale_factor = max_radius / end_radius_2d
        end.x *= scale_factor
        end.y *= scale_factor
    elif end_radius_2d < min_radius:
        scale_factor = min_radius / end_radius_2d
        end.x *= scale_factor
        end.y *= scale_factor

    # Update direction after constraints
    direction = end - start

    # Ensure branch angle doesn't create unprintable overhangs
    is_ok = adjust_z_value_if_angle_exceeds_threshold(direction)
    while not is_ok:
        direction.z += 0.1
        end = start + direction

        # Recalculate constraints
        max_radius = get_cone_radius(end.z, TREE_HEIGHT, BASE_RADIUS_OUTER)
        min_radius = get_cone_radius(end.z, INNER_CONE_HEIGHT, BASE_RADIUS_INNER)

        end_radius_2d = math.sqrt(end.x ** 2 + end.y ** 2)
        if end_radius_2d > max_radius:
            scale_factor = max_radius / end_radius_2d
            end.x *= scale_factor
            end.y *= scale_factor
        elif end_radius_2d < min_radius:
            scale_factor = min_radius / end_radius_2d
            end.x *= scale_factor
            end.y *= scale_factor
            
        direction = end - start
        is_ok = adjust_z_value_if_angle_exceeds_threshold(direction)

    # Create a cylinder for the branch
    mid = (start + end) / 2  # Cylinder midpoint
    bpy.ops.mesh.primitive_cylinder_add(
        radius=branch_radius, 
        depth=math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2) * DIRECTION_MODIFIER, 
        location=mid
    )
    branch_obj = bpy.context.object

    # Align the cylinder to the branch direction
    branch_obj.rotation_mode = 'QUATERNION'
    branch_obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')

    created_objects.append(branch_obj)

    # Determine number of child branches based on current depth
    if 2 < depth < DEPTH_FIRST_LIMIT:
        num_branches = first_phase_branches_chance()
    elif DEPTH_SECOND_LIMIT > depth > DEPTH_FIRST_LIMIT:
        num_branches = second_phase_branches_chance()
    else:
        num_branches = last_phase()

    child_objects = []

    # Create child branches
    for l in range(num_branches):
        # Calculate random angle for branching
        if depth < DEPTH_FIRST_LIMIT:
            random_angle = (1 - np.random.normal(loc=0, scale=std_dev)) * ANGLE_FIRST_PHASE * np.random.choice([1, -1])
        else:
            random_angle = (1 - np.random.normal(loc=0, scale=std_dev)) * ANGLE_SECOND_PHASE * np.random.choice([1, -1])

        if num_branches == 2:
            random_angle = clamp_angle_to_0_78_radians(random_angle)

        # Get surface normal for rotation axis
        surface_normal = get_cone_surface_normal(start)
        random_axis = surface_normal

        # Create rotation matrix and apply to direction
        rotation_matrix = mathutils.Matrix.Rotation(random_angle, 4, random_axis)
        new_dir = direction.copy()
        
        if num_branches == 1:
            direction *= SINGLE_EXTENDER
            
        new_dir = rotation_matrix @ new_dir

        # Create child branch with offset
        new_end = end.copy()
        new_end.z += Z_branch_offset * l
        child_obj = create_branch(new_end, new_dir, depth + 1, max_depth)
        
        if child_obj:
            child_objects.append(child_obj)

    # Join child branches to parent with proper topology
    if child_objects:
        upper_edge_coords = get_edge_loop(branch_obj, end.z)
        
        for k, child in enumerate(child_objects):
            print("Processing child branch")
            
            # Update the child object
            child.update_tag()
            child.data.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()

            # Get edge loop coordinates
            lower_edge_coords = get_edge_loop(child, end.z)
            print(f"Vertices in edge loops: {len(lower_edge_coords)}, {len(upper_edge_coords)}")
            print(f"Height of intersection: {end.z}")
            
            
            # Join the objects
            bpy.ops.object.select_all(action='DESELECT')
            branch_obj.select_set(True)
            child.select_set(True)
            bpy.context.view_layer.objects.active = branch_obj

            print(f"Origin before join: {branch_obj.location}")

            bpy.ops.object.join()
            branch_obj = bpy.context.view_layer.objects.active  # Update branch_obj reference

            # Update the created_objects list
            created_objects.remove(child)

            print(f"Origin after join: {branch_obj.location}")

            # Bridge edge loops to create smooth connections
            if lower_edge_coords and upper_edge_coords:
                print("Bridging edge loops")
                bpy.ops.object.mode_set(mode='EDIT')
                
                bm = bmesh.from_edit_mesh(bpy.context.object.data)
                tolerance = 1e-4  # Precision tolerance

                # Find vertices corresponding to the lower edge loop
                lower_edge_verts = []
                for v in bm.verts:
                    world_v_co = branch_obj.matrix_world @ v.co  # Transform to world space
                    for coord in lower_edge_coords:
                        if (world_v_co - coord).length < tolerance:
                            lower_edge_verts.append(v)
                            break

                # Handle upper edge vertices differently for first vs. subsequent children
                if k > 0:  # If not the first child
                    # Create new vertices based on existing edge loop coordinates
                    upper_edge_verts_copy = []
                    for coord in upper_edge_coords:
                        # Convert world coordinates to object coordinates
                        v_new = bm.verts.new(branch_obj.matrix_world.inverted() @ coord)
                        upper_edge_verts_copy.append(v_new)

                    # Create a face for the new edge loop if possible
                    if len(upper_edge_verts_copy) >= 3:
                        try:
                            bm.faces.new(upper_edge_verts_copy)
                        except ValueError:
                            print("Warning: face creation failed (likely non-planar edge loop)")

                    upper_edge_verts = upper_edge_verts_copy
                    bmesh.update_edit_mesh(bpy.context.object.data)
                else:
                    # Find vertices corresponding to the upper edge loop
                    upper_edge_verts = []
                    for v in bm.verts:
                        world_v_co = branch_obj.matrix_world @ v.co
                        for coord in upper_edge_coords:
                            if (world_v_co - coord).length < tolerance:
                                upper_edge_verts.append(v)
                                break
                    bmesh.update_edit_mesh(bpy.context.object.data)

                print(f"Vertices in edge loops: {len(lower_edge_verts)}, {len(upper_edge_verts)}")

                # Check for overlapping vertices
                if any(v in upper_edge_verts for v in lower_edge_verts):
                    print("ERROR: Lower and upper edge vertex lists share vertices!")
                    return None

                # Deselect all vertices
                if bpy.context.object and bpy.context.object.mode == 'EDIT':
                    bpy.ops.mesh.select_all(action='DESELECT')

                # Select vertices for bridging
                bpy.ops.mesh.select_mode(type="VERT")
                
                for v in lower_edge_verts:
                    v.select = True
                for v in upper_edge_verts:
                    v.select = True

                bmesh.update_edit_mesh(bpy.context.object.data)  # Update the mesh

                # Bridge the edge loops
                bpy.ops.mesh.bridge_edge_loops()
                bpy.ops.mesh.bridge_edge_loops()  # Sometimes needs to be called twice
                bpy.ops.object.mode_set(mode='OBJECT')  # Switch back to object mode
            else:
                print("Could not get edge loops for bridging")
    else:
        print("No child objects created")
        
    return branch_obj


def main() -> None:
    """
    Main function to generate the fractal tree structure and finalize the model.
    """
    # Generate initial branches from different starting points
    create_branch(mathutils.Vector((0.001, 0.0015, 0)), mathutils.Vector((1, 1.5, 1)), depth=0, max_depth=10)
    #create_branch(mathutils.Vector((-0.001, -0.0015, 0)), mathutils.Vector((-1.5, -1.5, 1)), depth=0, max_depth=15)
    #create_branch(mathutils.Vector((-0.001, -0.0015, 0)), mathutils.Vector((-1.5, -1, 1)), depth=0, max_depth=15)
    
    # Join all branches into a single mesh
    for obj in created_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = created_objects[0]
    bpy.ops.object.join()
    
    # Clean up the mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Position and rotate the model
    bpy.context.object.location.z = 20
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.ops.transform.rotate(value=math.radians(180), orient_axis='X')
    
    # Copy and rename pot objects
    copy_object_with_suffix("Bottom_to_copy")
    copy_object_with_suffix("TOP_to_copy")
    #copy_and_rename_pot("TOP_to_copy")
    
    # Join with pot objects
    join_with_pot("Bottom")
    join_with_pot("TOP")
    
    # Generate a UUID and rename the final object
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID for brevity
    bpy.context.object.name = f"outer_shell_{unique_id}"
    
    print(f"Fractal tree model '{bpy.context.object.name}' ready for 3D printing!")


def join_with_pot(pot_name: str) -> None:
    """
    Joins the tree structure with a pot object if it exists in the scene.
    
    Args:
        pot_name: Name of the pot object in the Blender scene
    """
    pot_obj = bpy.data.objects.get(pot_name)
    if pot_obj:
        pot_obj.select_set(True)
        bpy.ops.object.join()
        print(f"Joined with {pot_name} object")
    else:
        print(f"Object '{pot_name}' not found")


# Execute the main function when the script is run
if __name__ == "__main__":
    main()