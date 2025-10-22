import { PasswordTextField, TextField, ModalFooter } from "src/components";
import { FormProvider, useForm } from "react-hook-form";
import { Stack } from "@mui/material";
import * as yup from "yup";
import { yupResolver } from "@hookform/resolvers/yup";

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
});

const SignUpForm = ({ onClose }) => {
  const formMethods = useForm({
    defaultValues: { email: "", password: "", confirmPassword: "" },
    resolver: yupResolver(schema),
  });

  const { register, handleSubmit } = formMethods;

  //   TODO add signup logic
  const onSignUp = () => {};

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
      </FormProvider>
      <ModalFooter onClose={onClose} onConfirm={handleSubmit(onSignUp)} />
    </Stack>
  );
};

export default SignUpForm;
