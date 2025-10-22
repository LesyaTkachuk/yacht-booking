import { Stack } from "@mui/material";
import { styled, css } from "@mui/material/styles";

export const StyledStack = styled(Stack)(({ theme }) =>
  css({
    backgroundColor: theme.palette.special.white,
    padding: "24px 62px",
    justifyContent: "space-between",
    alignItems: "center",
    flexDirection: "row",
  })
);
