import { styled, css } from "@mui/material/styles";
import { Stack } from "@mui/material";

export const StyledWrapper = styled(Stack)(
  ({ theme }) => css`
    background-image: url("src/assets/images/harbour.jpg");
    background-size: cover;
    background-position: center;
    height: 100%;
    flex: 1;
    justify-content: center;
    align-items: center;
    min-height: calc(100vh - 98px);
  `
);
