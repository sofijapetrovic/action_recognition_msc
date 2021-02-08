from abc import ABC, abstractmethod
from skeleton.joint import JointType
import numpy as np


class Skeletons(ABC):
    def __init__(self, joints):
        self.joint_types = [j.value for j in JointType]
        self.joints_dict = {}
        self.limbs_dict = {}
        self.shape = None
        if joints is not None:
            self.group_joints(joints)

    def set_shape(self, shape):
        self.shape = shape

    def scale(self, output_shape):
        """
        Scale the joint coordinates from the input to the output shape
        Args:
            input_shape (tuple): input shape (height, width)
            output_shape (tuple): output shape (height, width)
        """
        for joint_type in self.joints_dict:
            for joint in self.joints_dict[joint_type]:
                joint.y *= output_shape[0] / self.shape[0]
                joint.y = int(joint.y)
                joint.x *= output_shape[1] / self.shape[1]
                joint.x = int(joint.x)
        self.shape = output_shape

    def get_hand_positions(self):
        """
        Get list of tuples of hand x and y positions
        """
        positions = []
        if JointType.LEFT_HAND in self.joints_dict.keys():
            for joint in self.joints_dict[JointType.LEFT_HAND]:
                positions.append((joint.x, joint.y))
        if JointType.RIGHT_HAND in self.joints_dict.keys():
            for joint in self.joints_dict[JointType.RIGHT_HAND]:
                positions.append((joint.x, joint.y))
        return positions

    @abstractmethod
    def connect_joints(self):
        pass

    def get_joints_dict(self):
        return self.joints_dict

    def group_joints(self, joints):
        """
        Groups joints by the joint type in the joint_dict
        Args:
            joints (list of Joint objects): all the joints to group
        """
        for joint in joints:
            if joint.type not in self.joints_dict.keys():
                self.joints_dict[joint.type] = [joint]
            else:
                self.joints_dict[joint.type].append(joint)

    def clear_unpaired_joints(self):
        """
        Makes a new joints dict containing only the joints that are a part of a limb in the limbs dict
        """
        self.joints_dict = {}
        joints = []
        for limb_type in self.limbs_dict.keys():
            for limb in self.limbs_dict[limb_type]:
                joint1, joint2 = limb.get_joints()
                joints.append(joint1)
                joints.append(joint2)
        filtered_joints = []
        [filtered_joints.append(joint) for joint in joints if joint not in filtered_joints]
        self.group_joints(filtered_joints)

    def draw(self, image):
        """
        Draws all the limbs contained in self.limbs_dict on the input image
        Args:
            image (np.array): raw image
        """
        if not self.limbs_dict:
            self.connect_joints()
        for limb_type in self.limbs_dict.keys():
            for limb in self.limbs_dict[limb_type]:
                limb.draw(image)
        for joint_type in self.joints_dict:
            for joint in self.joints_dict[joint_type]:
                joint.draw(image)


def assignment(cost):
    M, N = cost.shape
    assignment_array = [-1] * M
    if cost.size == 0:
        return assignment_array
    assigned = np.zeros(N)
    mins = np.min(cost, axis=1)
    order = np.argsort(mins)
    for cur_row in order:
        ordered_cols = np.argsort(cost[cur_row, :])
        for cur_col in ordered_cols:
            if assigned[cur_col]:
                continue
            assignment_array[cur_row] = cur_col
            assigned[cur_col] = 1
            break
    return assignment_array


def pair_joints(skeletons1, skeletons2):
    joint_pairs_dict = {}
    for joint_type in skeletons1.joints_dict.keys():
        if joint_type in skeletons2.joints_dict.keys():
            pairs = []
            joints1 = skeletons1.joints_dict[joint_type]
            joints2 = skeletons2.joints_dict[joint_type]
            M = len(joints1)
            N = len(joints2)
            if M>0 and N>0:
                cost = np.zeros((M, N))
                for i in range(M):
                    for j in range(N):
                        cost[i, j] = (joints1[i].x - joints2[j].x) ** 2 + (joints1[i].y - joints2[j].y) ** 2
                assignment_array = assignment(cost)
                for ind1 in range(M):
                    if assignment_array[ind1] > -1:
                        #print(assignment_array, ind1, M, N)
                        pairs.append((joints1[ind1], joints2[assignment_array[ind1]]))
            joint_pairs_dict[joint_type] = pairs
    return joint_pairs_dict
