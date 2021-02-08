from skeleton.skeletons import Skeletons
from skeleton.joint import Limb, LIMB_JOINT_DICT, LimbType, JointType, Joint
import numpy as np
import cv2


class AnnotatedSkeletons(Skeletons):
    def __init__(self, joints=None, joint_table=None, shape=(1080,1920), max_joints=40):
        if joints is None and joint_table is not None:
            joints = self.from_joint_table(joint_table)
        super().__init__(joints)
        self.max_joints = max_joints
        self.shape = shape
        self.person_ids = set([])
        self.connect_joints()

    def from_joint_table(self, joint_table):
        if not isinstance(joint_table, np.ndarray):
            joint_table = joint_table.detach().cpu().numpy()
        joints = []
        for i in range(joint_table.shape[0]):
            if joint_table[i, 0] == -1:
                break
            joints.append(Joint(x=joint_table[i, 0], y=joint_table[i, 1], type=JointType(joint_table[i, 2])))
        return joints

    def to_joint_table(self):
        joint_table = np.ones((self.max_joints, 3)) * -1
        ind = 0
        for joint_type in self.joints_dict.keys():
            for joint in self.joints_dict[joint_type]:
                joint_table[ind, 0] = joint.x
                joint_table[ind, 1] = joint.y
                joint_table[ind, 2] = joint_type.value
                ind += 1
        return joint_table

    def connect_joints(self):
        """
        Create and populate limbs_dict from the joints in joints_dict by connecting the ones with the same ID
        """
        for limb in LIMB_JOINT_DICT.keys():
            joint1_id = LIMB_JOINT_DICT[limb][0]
            joint2_id = LIMB_JOINT_DICT[limb][1]
            if joint1_id not in self.joints_dict.keys() or joint2_id not in self.joints_dict.keys():
                continue
            first_joint_candidates = self.joints_dict[joint1_id]
            second_joint_candidates = self.joints_dict[joint2_id]
            for joint1 in first_joint_candidates:
                for joint2 in second_joint_candidates:
                    self.person_ids.add(joint1.person_id)
                    self.person_ids.add(joint2.person_id)
                    if joint1.person_id == joint2.person_id:
                        new_limb = Limb(joint1, joint2, LimbType(joint1_id.value))
                        if joint1_id not in self.limbs_dict.keys():
                            self.limbs_dict[joint1_id] = [new_limb]
                        else:
                            self.limbs_dict[joint1_id].append(new_limb)

    def remove_out_of_bounds(self):
        """
        Remove the joints from the joints_dict that are out of the given image bounds
        Args:
            bounds_shape (tuple): bounds shape (height, width)
        """
        for joint_type in self.joints_dict:
            remove_joints = []
            for joint in self.joints_dict[joint_type]:
                if not (0 <= joint.x < self.shape[1] and 0 <= joint.y < self.shape[0]):
                    remove_joints.append(joint)
            for joint in remove_joints:
                self.joints_dict[joint_type].remove(joint)

    def to_list(self):
        """
        Creates list of tuples (x,y) from all the saved joints in joint_dict
        Returns:
            coord_list (list): List of tuples of joint coordinates
        """
        coord_list = []
        for joint_type in self.joints_dict:
            for joint in self.joints_dict[joint_type]:
                coord_list.append((joint.x, joint.y))
        return coord_list

    def from_list(self, coord_list):
        """
        Updates the joint coordinates from the joint_dict using the given coordinate list
        Args:
            coord_list (list): list of tuples (x,y) representing the joint coordinates
        """
        ind = 0
        for joint_type in self.joints_dict:
            for joint in self.joints_dict[joint_type]:
                joint.x = int(coord_list[ind][0])
                joint.y = int(coord_list[ind][1])
                ind += 1

    def get_keypoint_map(self, sigma):
        """
        Generates joint heatmap of all the joints in the joint_dict
        Args:
            sigma (float): sigma for the gaussian blur

        Returns:
            head_map (np.array): heat map image in shape (width, height, joint_type_num)
        """
        image = np.zeros((self.shape[0], self.shape[1], len(JointType)))
        for joint_type in self.joints_dict.keys():
            for joint in self.joints_dict[joint_type]:
                layer = np.zeros((self.shape[0], self.shape[1]))
                layer[joint.y, joint.x] = 1
                layer = cv2.GaussianBlur(layer, (0, 0), sigma)
                layer = layer/np.max(layer)
                image[:, :, joint_type.value] = np.maximum(image[:, :, joint_type.value], layer)

        return image

    def get_paf_map(self, dist):
        image = np.zeros((self.shape[0], self.shape[1], len(LimbType)*2))
        people_no = len(self.person_ids)
        for limb_type in self.limbs_dict.keys():
            for limb in self.limbs_dict[limb_type]:
                ang = np.arctan2(limb.joint2.y - limb.joint1.y, limb.joint2.x - limb.joint1.x)
                v = (np.cos(ang), np.sin(ang))
                image[:, :, limb_type.value * 2] += cv2.line(np.zeros_like(image[:, :, 0]),
                                                               (limb.joint1.x, limb.joint1.y),
                                                               (limb.joint2.x, limb.joint2.y),
                                                               color=v[0],
                                                               thickness=dist) / people_no
                image[:, :, limb_type.value * 2 + 1] += cv2.line(np.zeros_like(image[:, :, 1]),
                                                                   (limb.joint1.x, limb.joint1.y),
                                                                   (limb.joint2.x, limb.joint2.y),
                                                                   color=v[1],
                                                                   thickness=dist) / people_no
        return image


