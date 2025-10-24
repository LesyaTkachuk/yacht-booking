import { useState, useEffect } from "react";
import { NavBar, UserMenu, AuthButtons } from "src/components";

import LogoIcon from "src/assets/logo.svg";
import { StyledStack } from "./styled";

const Header = () => {
  const [token, setToken] = useState(localStorage.getItem("token"));

  useEffect(() => {
    (async () => {
      const storedToken = localStorage.getItem("token");
      if (storedToken) {
        setToken(storedToken);
      }
    })();
  });
  return (
    <StyledStack>
      <LogoIcon />
      <NavBar />
      {token ? <UserMenu /> : <AuthButtons />}
    </StyledStack>
  );
};

export default Header;
