import MuiButton from "@mui/material/Button";
import { css, styled } from "@mui/material/styles";

export const StyledButton = styled(MuiButton)(
  ({ fullWidth }) => css`
    font-size: 1rem;
    line-height: 1.5rem;
    width: ${fullWidth ? "100%" : "fit-content"};
    height: fit-content;
  `
);
