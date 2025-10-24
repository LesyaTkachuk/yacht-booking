import { ROUTES } from "src/navigation/routes";
import { useLocation } from "react-router-dom";
import { StyledNav, StyledLink } from "./styled";

const NavBar = () => {

  return (
    <StyledNav>
      <StyledLink to="/">Home</StyledLink>
      <StyledLink to={ROUTES.ABOUT}>About</StyledLink>
    </StyledNav>
  );
};

export default NavBar;
