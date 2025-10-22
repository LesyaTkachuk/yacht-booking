import { Typography } from "@mui/material";
import { Button, Modal, SignInForm, SignUpForm } from "src/components";

const AuthModal = ({
  isLoginModalOpen,
  isRegisterModalOpen,
  openLoginModal,
  closeLoginModal,
  openRegisterModal,
  closeRegisterModal,
}) => {
  const onSwitchToLogin = () => {
    openLoginModal();
    closeRegisterModal();
  };

  const onSwitchToRegister = () => {
    openRegisterModal();
    closeLoginModal();
  };

  return (
    <>
      {isLoginModalOpen && (
        <Modal open title={"SIGN IN"} onClose={closeLoginModal}>
          <SignInForm onClose={closeLoginModal} />
          <Typography>
            Don&apos;t have an account?
            <Button variant="text" onClick={onSwitchToRegister}>
              Create an account
            </Button>
          </Typography>
        </Modal>
      )}
      {isRegisterModalOpen && (
        <Modal open title={"SIGN UP"} onClose={closeRegisterModal}>
          <SignUpForm onClose={closeRegisterModal} />
          <Typography>
            Already have an account?
            <Button variant="text" onClick={onSwitchToLogin}>
              Sign In
            </Button>
          </Typography>
        </Modal>
      )}
    </>
  );
};

export default AuthModal;
