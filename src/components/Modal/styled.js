import { css, Modal, Stack, styled } from "@mui/material";

export const Content = styled(Stack)`
  position: relative;
  width: ${({ width }) => width ?? "520px"};
  height: auto;
  min-height: 20%;
  max-height: 90%;
  background-color: white;
  outline: none;
  border-radius: 16px;
  align-items: center;
  justify-content: space-between;
  padding: 24px;
  padding-right: 0;
`;

export const StyledModal = styled(Modal)`
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const StyledFooterContainer = styled(Stack)(
  ({ theme }) => css`
    position: sticky;
    bottom: 0;
    background-color: ${theme.palette.special.white};
    flex-direction: row;
    flex: 1;
    padding-top: 20px;
    width: 100%;
    gap: 10px;
  `
);
