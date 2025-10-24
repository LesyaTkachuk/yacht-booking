import Button from "../Button";

import { StyledFooterContainer } from "./styled";

const ModalFooter = ({
  onClose,
  onConfirm,
  isPending = false,
  confirmBtnTitle = "Save",
  confirmBtnColor = "primary",
}) => {
  return (
    <StyledFooterContainer>
      <Button
        fullWidth
        variant="outlined"
        color="primary"
        size="large"
        onClick={onClose}
      >
        Cancel
      </Button>
      {onConfirm && (
        <Button
          fullWidth
          size="large"
          isLoading={isPending}
          onClick={onConfirm}
          color={confirmBtnColor}
        >
          {confirmBtnTitle}
        </Button>
      )}
    </StyledFooterContainer>
  );
};

export default ModalFooter;
