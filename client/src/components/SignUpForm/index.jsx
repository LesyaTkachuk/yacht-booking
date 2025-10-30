import {
  PasswordTextField,
  TextField,
  Select,
  ModalFooter,
  RadioGroup,
  showError,
  showSuccess,
} from "src/components";
import { FormProvider, useForm } from "react-hook-form";
import { Stack } from "@mui/material";
import * as yup from "yup";
import { yupResolver } from "@hookform/resolvers/yup";
import { USER_ROLES, SAILING_EXPERIENCE } from "src/constants/user";
import { registerUser } from "src/services/auth";
import { useMutation } from "@tanstack/react-query";
import { capitalizeFirstLetter, isValidNumber } from "src/utils";

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
    defaultValues: {
      email: "",
      password: "",
      confirmPassword: "",
      role: "",
      country: "",
      sailingExp: "",
      budgetMin: "",
      budgetMax: "",
      hasSkipperLicense: false,
    },
    resolver: yupResolver(schema),
  });

  const roleItems = Object.values(USER_ROLES).map((value) => ({
    value,
    title:
      value === USER_ROLES.LESSER
        ? "I have a yacht for rent"
        : "I want to rent a yacht",
  }));

  const sailingExperienceItems = Object.values(SAILING_EXPERIENCE).map(
    (value) => ({
      value,
      title: capitalizeFirstLetter(value),
    })
  );

  const hasSkipperLicenseItems = [
    {
      value: true,
      title: "Yes",
    },
    {
      value: false,
      title: "No",
    },
  ];

  const { register, handleSubmit, watch } = formMethods;

  const role = watch("role");

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
    const { confirmPassword, budgetMax, budgetMin, ...rest } = data;
    const numberBudgetMin = isValidNumber(budgetMin) ? Number(budgetMin) : null;
    const numberBudgetMax = isValidNumber(budgetMax) ? Number(budgetMax) : null;
    if (!numberBudgetMin) {
      showError(_, "Minimum budget must be a number");
      return;
    }
    if (!numberBudgetMax) {
      showError(_, "Maximum budget must be a number");
      return;
    }
    registerMutation({
      ...rest,
      budgetMin: numberBudgetMin,
      budgetMax: numberBudgetMax,
    });
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
        <Stack direction={"row"} gap={2}>
          <PasswordTextField
            width="50%"
            label="Password"
            {...register("password")}
            tooltipTitle="Enter a password"
          />
          <PasswordTextField
            width="50%"
            label="Confirm Password"
            {...register("confirmPassword")}
            tooltipTitle="Confirm a password"
          />
        </Stack>
        <RadioGroup
          formMethods={formMethods}
          items={roleItems}
          groupLabel="Role"
          name="role"
          tooltipTitle="Select the appropriate role based on your interaction with the platform."
        />
        {role === USER_ROLES.LESSEE && (
          <>
            <Stack direction={"row"} gap={2}>
              <TextField
                width="100%"
                label="Sailing Country"
                {...register("country")}
                tooltipTitle="Enter the country you would like to sail in."
              />
              <Select
                items={sailingExperienceItems}
                name="sailingExp"
                label={"Sailing Experience"}
              />
            </Stack>
            <Stack direction={"row"} gap={2}>
              <TextField
                isNumeric
                width="50%"
                label="Minimum Budget"
                {...register("budgetMin")}
                tooltipTitle="Enter the minimum budget per day you are looking for."
              />
              <TextField
                isNumeric
                width="50%"
                label="Maximum Budget"
                {...register("budgetMax")}
                tooltipTitle="Enter the maximum budget per day you are looking for."
              />
            </Stack>
            <RadioGroup
              formMethods={formMethods}
              items={hasSkipperLicenseItems}
              groupLabel="Has Skipper License?"
              name="hasSkipperLicense"
              tooltipTitle="Select if you have a skipper license."
            />
          </>
        )}
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
