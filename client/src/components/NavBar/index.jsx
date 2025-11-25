import { ROUTES } from "src/navigation/routes";
import { useAuth } from "src/context/authContext";
import { USER_ROLES } from "src/constants/user";
import { StyledNav, StyledLink } from "./styled";

const NavBar = () => {
  const { user } = useAuth();

  return (
    <StyledNav>
      <StyledLink to="/">Home</StyledLink>
      <StyledLink to={ROUTES.ABOUT}>About</StyledLink>
      
      {user?.role === USER_ROLES.LESSER && (
        <StyledLink to={ROUTES.ADD_YACHT}>Add Yacht</StyledLink>
      )}
    </StyledNav>
  );
};

export default NavBar;
