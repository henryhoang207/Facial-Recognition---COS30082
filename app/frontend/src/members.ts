import type { ComponentType } from "react";
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
  };
}

// Thêm tính năng cá nhân: tạo component trong src/individual/ rồi gắn vào `feature` của người đó
export const MEMBERS: Member[] = [
  { id: "hoan", name: "Hoàng Hữu Hoan", short: "Hữu Hoan", core: "Anti-spoofing" },
  { id: "giakhanh", name: "Giáp Hoàng Gia Khánh", short: "Gia Khánh", core: "Face recognition, Model A" },
  { id: "ducanh", name: "Lê Đức Anh", short: "Đức Anh", core: "Face recognition, Model B" },
  { id: "manhduc", name: "Trần Mạnh Đức", short: "Mạnh Đức", core: "Emotion detection" },
  { id: "ducthanh", name: "Lê Đức Thành", short: "Đức Thành", core: "Datasets and shared face crop" },
  {
    id: "namkhanh",
    name: "Nguyễn Nam Khánh",
    short: "Nam Khánh",
    core: "Web app and integration",
    feature: {
      title: "Face mask detection",
      summary: "Tells, for every face in view, whether the person is wearing a face mask.",
      component: MaskDetection,
    },
  },
];
