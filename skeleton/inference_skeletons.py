import numpy as np
from skeleton.joint import Limb, LIMB_JOINT_DICT, LimbType
from skeleton.skeletons import Skeletons
import cv2

class InferenceSkeletons(Skeletons):
    def __init__(self, joints, paf_map, output_shape, sum_threshold=0.05):
        super().__init__(joints)
        self.shape = output_shape
        self.paf_map = paf_map
        self.sum_threshold = sum_threshold

    def integrate(self, paf_map, joint1, joint2, sample_dist=1):
        """
        Integrates the paf map along the line connecting the joints
        Args:
            paf_map (np.array): Paf map of the limb
            joint1 (skeleton.joint.Joint): Joint object of the first joint
            joint2 (skeleton.joint.Joint): Joint object of the second joint
            sample_dist (int): Integration step in pixels

        Returns: Int - total integrated sum

        """
        y1 = int(joint1.y * paf_map.shape[1] / self.shape[0])
        y2 = int(joint2.y * paf_map.shape[1] / self.shape[0])
        x1 = int(joint1.x * paf_map.shape[2] / self.shape[1])
        x2 = int(joint2.x * paf_map.shape[2] / self.shape[1])
        limb_len = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        ang = np.arctan2(y2 - y1, x2 - x1)
        sum = 0
        ymin = min(y1,y2)
        ymax = max(y1,y2)
        coef = 1
        for i in range(ymin, ymax, sample_dist):
            j = int((i - y1) * (x2 - x1) / (y2 - y1) + x1)
            paf_norm = np.sqrt(paf_map[0, i, j] ** 2 + paf_map[1, i, j] ** 2)
            paf_ang = np.arctan2(paf_map[1, i, j], paf_map[0, i, j])
            ang_diff = paf_ang - ang
            sum += paf_norm * np.cos(ang_diff)
        return sum * coef / limb_len

    def assignment(self, sums, sum_threshold):
        """
        Do greedy optimization for connecting joints into the limb
        Args:
            sums (np.array): integral sums corresponding to the pair of joints returned by the integrate() function
            sum_threshold (float): minimum sum to be able to connect two joints into the limb

        Returns: Assignment array of length sums.shape[0] where each element of the array corresponds to the index
        of column that should be matched with row of element's index to achieve the best match

        """
        M, N = sums.shape
        assignment_array = [-1] * M
        if sums.size == 0:
            return assignment_array
        assigned = np.zeros(N)
        maxs = np.max(sums, axis=1)
        order = np.argsort(-maxs)
        for cur_row in order:
            ordered_cols = np.argsort(-sums[cur_row, :])
            for cur_col in ordered_cols:
                if assigned[cur_col]:
                    continue
                if sums[cur_row][cur_col] < sum_threshold:
                    break
                assignment_array[cur_row] = cur_col
                assigned[cur_col] = 1
                break
        return assignment_array

    def connect_joints(self):
        """
        Matches the joints in self.joint_dict into the limbs by doing the integration of the corresponding paf maps and
        populates the self.limbs_dict
        Args:
            paf_map (np.array): MxNx2C image of the paf map (M - original image height, N - original image width, C -
            number of the limb types)
            sum_threshold (float): Threshold of the minimal integral sum that can correspond to the limb

        """
        if self.paf_map[0,:,:].shape != self.shape:
            new_pafmap = np.zeros((self.paf_map.shape[0], self.shape[0], self.shape[1]))
            for i in range(self.paf_map.shape[0]):
                new_pafmap[i,:,:] = cv2.resize(self.paf_map[i,:,:], (self.shape[1],self.shape[0]))
            self.paf_map = new_pafmap
        for limb in LIMB_JOINT_DICT.keys():
            joint1_id = LIMB_JOINT_DICT[limb][0]
            joint2_id = LIMB_JOINT_DICT[limb][1]
            if joint1_id not in self.joints_dict.keys() or joint2_id not in self.joints_dict.keys():
                continue
            first_joint_candidates = self.joints_dict[joint1_id]
            second_joint_candidates = self.joints_dict[joint2_id]
            sums = np.zeros((len(first_joint_candidates), len(second_joint_candidates)))
            for i, joint1 in enumerate(first_joint_candidates):
                for j, joint2 in enumerate(second_joint_candidates):
                    paf = self.paf_map[joint1_id.value*2:joint1_id.value*2+2, :, :]
                    sums[i, j] = self.integrate(paf, joint1, joint2)
            assignment_array = self.assignment(sums, self.sum_threshold)
            for first_joint_ind, second_joint_ind in enumerate(assignment_array):
                if second_joint_ind == -1:
                    continue
                new_limb = Limb(first_joint_candidates[first_joint_ind], second_joint_candidates[second_joint_ind],
                                LimbType(joint1_id.value))
                if first_joint_ind not in self.limbs_dict.keys():
                    self.limbs_dict[first_joint_ind] = [new_limb]
                else:
                    self.limbs_dict[first_joint_ind].append(new_limb)

