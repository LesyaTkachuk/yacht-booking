import { NavLink } from "react-router-dom";
import { styled, css } from "@mui/material/styles";

export const StyledNav = styled("nav")`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 10px;
  font-size: 18px;
  gap: 16px;
`;

export const StyledLink = styled(NavLink)(
  ({ theme }) => css`
    text-decoration: none;
    color: inherit;

    &.active {
      color: ${theme.palette.primary.main};
      font-weight: 600;
      text-decoration: underline;
    }

    &:hover {
      color: ${theme.palette.primary.main};
    }
  `
);
