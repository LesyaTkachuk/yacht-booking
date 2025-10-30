import { useEffect } from "react";
import { NavBar, UserMenu, AuthButtons } from "src/components";
import { useAuth } from "src/context/authContext";

import LogoIcon from "src/assets/logo.svg";
import { StyledStack } from "./styled";

const Header = () => {
  const { user } = useAuth();

  return (
    <StyledStack>
      <LogoIcon />
      <NavBar />
      {user?.token ? <UserMenu /> : <AuthButtons />}
    </StyledStack>
  );
};

export default Header;
