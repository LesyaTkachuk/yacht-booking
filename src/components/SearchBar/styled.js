import { styled, css } from "@mui/material/styles";

export const BorderedContainer = styled("div")(({ theme }) =>
  css({
    backgroundColor: theme.palette.special.white,
    width: "100%",
    border: "1px solid #000000",
    borderRadius: "16px",
    padding: "8px",
  })
);
