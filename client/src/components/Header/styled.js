import { Stack } from "@mui/material";
import { styled, css } from "@mui/material/styles";

export const StyledStack = styled(Stack)(({ theme }) =>
  css({
    position: "sticky",
    top: 0,
    zIndex: 100,
    backgroundColor: theme.palette.special.white,
    padding: "24px 62px",
    justifyContent: "space-between",
    alignItems: "center",
    flexDirection: "row",
  })
);
