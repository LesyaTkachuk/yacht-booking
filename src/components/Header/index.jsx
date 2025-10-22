import { NavBar, UserMenu, AuthButtons } from "src/components";

import LogoIcon from "src/assets/logo.svg";
import { StyledStack } from "./styled";

const Header = () => {
  return (
    <StyledStack>
      <LogoIcon />
      <NavBar />
      <AuthButtons />
      {/* TODO check if user is logged in */}
      {/* <UserMenu /> */}
    </StyledStack>
  );
};

export default Header;
