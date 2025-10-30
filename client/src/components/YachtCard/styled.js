import { styled, css } from "@mui/material/styles";
import { IconButton } from "@mui/material";
import { Button } from "src/components";

export const StyledListItem = styled("li")`
  cursor: pointer;
  width: 530px;
  gap: 12px;
  padding: 0;
  display: flex;
  flex-direction: column;
`;

export const StyledImageWrapper = styled("div")`
  position: relative;
  width: 530px;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 8px;

  &:hover {
    scale: 1.02;
    transition: all 0.3s ease-in-out;
  }
`;

export const StyledImage = styled("img")`
  width: 100%;
  height: auto;
  object-fit: cover;
  border-radius: 8px;
`;

export const StyledIconButton = styled(IconButton)`
  position: absolute;
  bottom: 8px;
  right: 8px;
  cursor: pointer;
`;
