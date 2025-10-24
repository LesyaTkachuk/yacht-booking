import { Stack } from "@mui/material";
import { Button, AuthModal } from "src/components";
import { useAuthHook } from "src/hooks";

const AuthButtons = () => {
  const {
    isLoginModalOpen,
    isRegisterModalOpen,
    openLoginModal,
    closeLoginModal,
    openRegisterModal,
    closeRegisterModal,
  } = useAuthHook();

  return (
    <>
      <Stack gap={1} direction={"row"}>
        <Button variant="outlined" onClick={openLoginModal}>
          Sign in
        </Button>
        <Button variant="outlined" onClick={openRegisterModal}>
          Sign up
        </Button>
      </Stack>
      <AuthModal
        isLoginModalOpen={isLoginModalOpen}
        isRegisterModalOpen={isRegisterModalOpen}
        openLoginModal={openLoginModal}
        closeLoginModal={closeLoginModal}
        openRegisterModal={openRegisterModal}
        closeRegisterModal={closeRegisterModal}
      />
    </>
  );
};

export default AuthButtons;
