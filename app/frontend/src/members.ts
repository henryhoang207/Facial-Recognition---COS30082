import type { ComponentType } from "react";
import type { IndividualMode } from "./types";
import { AdaptiveChallenge } from "./individual/AdaptiveChallenge";
import { AttendanceMLP } from "./individual/AttendanceMLP";
import { FaceParsing } from "./individual/FaceParsing";
import { HandGesture } from "./individual/HandGesture";
import { HeadPose } from "./individual/HeadPose";
import { MaskDetection } from "./individual/MaskDetection";

export interface Member {
  id: string;
  name: string;
  short: string; // tên hiện trên tab
  core: string; // phần việc chính, theo bảng Team trong README
  feature?: {
    title: string;
    summary: string;
    component: ComponentType;
    model: "mask" | "attendance" | IndividualMode; // để tab biết model đã chạy chưa
  };
}

// Thêm tính năng cá nhân: tạo component trong src/individual/ rồi gắn vào `feature` của người đó
export const MEMBERS: Member[] = [
  {
    id: "hoan",
    name: "Hoàng Hữu Hoan",
    short: "Hữu Hoan",
    core: "Anti-spoofing",
    feature: {
      title: "Adaptive liveness check",
      summary: "Trusts a clearly live face at once; a doubtful face must turn left and right before it is recognised.",
      component: AdaptiveChallenge,
      model: "challenge",
    },
  },
  {
    id: "giakhanh",
    name: "Giáp Hoàng Gia Khánh",
    short: "Gia Khánh",
    core: "Face recognition, Model A",
    feature: {
      title: "Attendance MLP",
      summary: "Uses five live frames from Model A to verify an identity before recording attendance.",
      component: AttendanceMLP,
      model: "attendance",
    },
  },
  {
    id: "ducanh",
    name: "Lê Đức Anh",
    short: "Đức Anh",
    core: "Face recognition, Model B",
    feature: {
      title: "ASL hand signs",
      summary: "Reads moving ASL signs (J, Z, Hello, Sorry, Thank you, Yes, No) from one hand.",
      component: HandGesture,
      model: "gesture",
    },
  },
  {
    id: "manhduc",
    name: "Trần Mạnh Đức",
    short: "Mạnh Đức",
    core: "Emotion detection",
    feature: {
      title: "Face parsing",
      summary: "Colours each part of the face: skin, brows, eyes, nose, lips, mouth and hair.",
      component: FaceParsing,
      model: "face_parsing",
    },
  },
  {
    id: "ducthanh",
    name: "Lê Đức Thành",
    short: "Đức Thành",
    core: "Datasets and shared face crop",
    feature: {
      title: "Head pose",
      summary: "Measures which way the head is turned: yaw, pitch and roll.",
      component: HeadPose,
      model: "head_pose",
    },
  },
  {
    id: "namkhanh",
    name: "Nguyễn Nam Khánh",
    short: "Nam Khánh",
    core: "Web app and integration",
    feature: {
      title: "Face mask detection",
      summary: "Tells, for every face in view, whether the person is wearing a face mask.",
      component: MaskDetection,
      model: "mask",
    },
  },
];
