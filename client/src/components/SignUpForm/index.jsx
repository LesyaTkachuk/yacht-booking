import {
  PasswordTextField,
  TextField,
  ModalFooter,
  RadioGroup,
  showError,
  showSuccess,
} from "src/components";
import { FormProvider, useForm } from "react-hook-form";
import { Stack } from "@mui/material";
import * as yup from "yup";
import { yupResolver } from "@hookform/resolvers/yup";
import { USER_ROLES } from "src/constants/user";
import { registerUser } from "src/services/auth";
import { useMutation } from "@tanstack/react-query";

const schema = yup.object().shape({
  email: yup.string().email("Invalid email").required("Email is required"),
  password: yup
    .string()
    .min(8, "Password must be at least 8 characters")
    .required("Password is required"),
  confirmPassword: yup
    .string()
    .oneOf([yup.ref("password")], "Passwords must match")
    .required("Please confirm your password"),
  role: yup
    .string()
    .oneOf(["lesser", "lessee"], "Role must be oneof the specified values")
    .required("Role is required"),
});

const SignUpForm = ({ onClose }) => {
  const formMethods = useForm({
    defaultValues: { email: "", password: "", confirmPassword: "", role: "" },
    resolver: yupResolver(schema),
  });

  const roleItems = Object.values(USER_ROLES).map((value) => ({
    value,
    title:
      value === USER_ROLES.LESSER
        ? "I have a yacht for rent"
        : "I want to rent a yacht",
  }));

  const { register, handleSubmit } = formMethods;

  const { mutate: registerMutation, isPending } = useMutation({
    mutationFn: registerUser,
    onSuccess: () => {
      onClose();
      showSuccess(
        "User is successfully registered. Verification email sent. Please check your inbox and follow the instructions. Once verified, you can log in."
      );
    },

    onError: (e) => {
      showError(
        e,
        "There was an error registering your account. Please try again."
      );
    },
  });

  const onSignUp = async (data) => {
    const { confirmPassword, ...rest } = data;
    registerMutation(rest);
  };

  // TODO add lessee specific fields budget, sailing experience
  return (
    <Stack gap={2} width={"76%"}>
      <FormProvider {...formMethods}>
        <TextField
          width="100%"
          label="Email"
          {...register("email")}
          tooltipTitle="Enter an email address to sign in"
        />
        <PasswordTextField
          width="100%"
          label="Password"
          {...register("password")}
          tooltipTitle="Enter a password"
        />
        <PasswordTextField
          width="100%"
          label="Confirm Password"
          {...register("confirmPassword")}
          tooltipTitle="Enter a password"
        />
        <RadioGroup
          formMethods={formMethods}
          items={roleItems}
          groupLabel="Role"
          name="role"
          tooltipTitle="Select the appropriate role based on your interaction with the platform."
        />
      </FormProvider>
      <ModalFooter
        onClose={onClose}
        onConfirm={handleSubmit(onSignUp)}
        confirmBtnTitle="Sign Up"
        isPending={isPending}
      />
    </Stack>
  );
};

export default SignUpForm;
