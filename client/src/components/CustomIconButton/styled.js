import { css, IconButton, styled, alpha } from "@mui/material";

export const StyledIconButton = styled(IconButton)(
  ({ theme }) => css`
    width: fit-content;
    height: fit-content;
    padding: 6px;

    &:hover {
      background-color: ${alpha(theme.palette.primary.main, 0.1)};
    }
  `
);
