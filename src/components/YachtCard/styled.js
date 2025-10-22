import { styled, css } from "@mui/material/styles";
import { IconButton, Stack } from "@mui/material";

export const StyledWrapper = styled(Stack)(
  ({ theme }) => css`
    width: 100%;
    gap: 16px;
  `
);

export const StyledImageWrapper = styled("div")(
  ({ theme }) => css`
    position: relative;
    width: 650px;
    height: 550px;
    border-radius: 16px;
    overflow: hidden;
  `
);

export const StyledImage = styled("img")(
  ({ theme }) => css`
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 16px;
  `
);

export const StyledIconButton = styled(IconButton)`
  position: absolute;
  bottom: 8px;
  right: 8px;
  cursor: pointer;
`;
