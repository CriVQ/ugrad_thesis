using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using System.Linq;
using UnityEngine.SceneManagement;

public class EMGInteraction : MonoBehaviour
{

    //[SerializeField]  XRGrabInteractable grabbable;

    XRBaseInteractable grabbable;

    private XRDirectInteractor handInteractor;
    private GameObject rightHandController;
    private GameObject leftHandController;
    HandComplete righthandcomplete;
    HandComplete lefthandcomplete;

    public Hovercheck righthovercheck;
    public Hovercheck lefthovercheck;
    public bool _ManualAnimate = false;


    [SerializeField] private GameObject CanvasCalib;

    public ArduinoCom _arduino;

    public enum Hand { Right, Left };
    public Hand CurrentHand { get; private set; }


    public int EMGThreshold = 15;
    public float CalibrationDuration = 3f;

    public bool _EMGDebugLog = false;
    private bool feedbackState = false;
    private bool _initialHandSelect = false;



    private void Start()
    {
        rightHandController = GameObject.Find("RightHand Controller");
        leftHandController = GameObject.Find("LeftHand Controller");

        righthandcomplete = rightHandController.GetComponent<HandComplete>();
        lefthandcomplete = leftHandController.GetComponent<HandComplete>();

        righthovercheck = rightHandController.GetComponent<Hovercheck>();
        lefthovercheck = leftHandController.GetComponent<Hovercheck>();

        SetLeftHand();
        SetLeftHand();
        SetRightHand();
        _initialHandSelect = true;

        turnoffvibration();

    }

    private void FixedUpdate()
    {

        float input = _arduino.GetData();
        
        
        if (_EMGDebugLog == true)
        Debug.Log("EMG:\t" + input);

        if (input >= EMGThreshold)
        {
            HandleFlex();
            if (CurrentHand == Hand.Right)
                righthandcomplete.HideHandOnSelectEMG();
            if (CurrentHand == Hand.Left)
                lefthandcomplete.HideHandOnSelectEMG();

            if (!feedbackState)
            {
                _arduino.SendData("1");
                Invoke("turnoffvibration", 0.05f);
                feedbackState = true;

            }
        }
        else
        {
            HandleRelaxed();

            if (!_ManualAnimate)
            {

                if (CurrentHand == Hand.Right)
                    righthandcomplete.HideHandOnDeSelectEMG();
                if (CurrentHand == Hand.Left)
                    lefthandcomplete.HideHandOnDeSelectEMG();
                feedbackState = false;
            }
        }

       
    }

    public float LiveEMGValue(float liveEMG)
    {
        liveEMG = _arduino.GetData();
        return liveEMG;
    }


    private void Update() //All Keyboard Inputs only
    {
        if (Input.GetKeyDown(KeyCode.C))
            GetAvgValue();

        if (Input.GetKeyDown(KeyCode.L))
            SetLeftHand();

        if (Input.GetKeyDown(KeyCode.R))
            SetRightHand();

        if (Input.GetKeyDown(KeyCode.Q))
            SceneManager.LoadScene(12);

    }


    private void HandleFlex()
    {
        if (grabbable = null)
        {
            Debug.Log("Not hovering over interactable");
        }

        if (!handInteractor.isSelectActive && (righthovercheck._isHovering == true || lefthovercheck._isHovering == true ))
        {
            

            if (CurrentHand == Hand.Right && righthovercheck._isHovering == true)
            {
                righthandcomplete.HideHandOnSelectEMG();
                grabbable = righthovercheck._interactableobject;

                Debug.Log(grabbable);

            }
            else if (CurrentHand == Hand.Left && lefthovercheck._isHovering == true)
            {

                lefthandcomplete.HideHandOnSelectEMG();
                grabbable = lefthovercheck._interactableobject;
            }

            handInteractor.StartManualInteraction(grabbable);

        }

    }


    private void HandleRelaxed()
    {
        // if (handInteractor.isSelectActive)
        if(handInteractor.isPerformingManualInteraction)
        {
            

            handInteractor.EndManualInteraction();

            if (CurrentHand == Hand.Right)
            {
                if(!_ManualAnimate)
                righthandcomplete.HideHandOnDeSelectEMG();
            }
            else if (CurrentHand == Hand.Left)
            {
                if(!_ManualAnimate)
                lefthandcomplete.HideHandOnDeSelectEMG();
            }

        }

    }






    #region Seperate settings for Right & Left Hands

    public void SetRightHand()
    {
      
        {

            if (_initialHandSelect == true)
            {
                handInteractor.EndManualInteraction();
            }

            CanvasCalib.SetActive(true);
            UIText.instance.Right();

            handInteractor = rightHandController.GetComponent<XRDirectInteractor>();
            CurrentHand = Hand.Right;

            righthandcomplete.SettoRobot();
            lefthandcomplete.SettoHuman();
  
            Debug.Log("RightHandSelected");
            Invoke("DisableCanvas", 1f);
        }
    }

    public void SetLeftHand()
    {
     //   if (CurrentHand != Hand.Left)
        {

            if (_initialHandSelect == true)
            {
                handInteractor.EndManualInteraction();
            }

            CanvasCalib.SetActive(true);
            UIText.instance.Left();


            handInteractor = leftHandController.GetComponent<XRDirectInteractor>();
            CurrentHand = Hand.Left;


            lefthandcomplete.SettoRobot();
            righthandcomplete.SettoHuman();
            
            Debug.Log("LeftHandSelected");
            Invoke("DisableCanvas", 1f);

        }
        

    }

    #endregion

    #region Calibration stuff

    public void GetAvgValue()
    {
        List<float> data = new List<float>();
        float averageData = 0f;

        UIText.instance.Calibrating();
         CanvasCalib.SetActive(true);
        StartCoroutine(CollectDataCoroutine(data, () =>
        {
            if (data.Count > 0)
            {
                averageData = data.Average();
            }

            Debug.Log("Average Value While Flexing = " + averageData);
            EMGThreshold = (int)averageData;
           CanvasCalib.SetActive(false);
        }));
    }

    private IEnumerator CollectDataCoroutine(List<float> data, Action onComplete)
    {
        float elapsedTime = 0f;
        // float CalibrationDuration = 3f;
        float startTime = Time.time;

        while (elapsedTime < CalibrationDuration)
        {
            Debug.Log("Calibrating Flex Value........");
            float input = _arduino.GetData();
            data.Add(input);
            elapsedTime = Time.time - startTime;
            yield return null; // Wait for the next frame
        }

        onComplete?.Invoke(); // Invoke the callback function
    }
    #endregion

    private void DisableCanvas()
    {
        CanvasCalib.SetActive(false);
    }
    private void turnoffvibration()
    {
        //serial.Write("0");
        _arduino.SendData("0");

    }

    public void AnimateFist()
    {
        if (CurrentHand == Hand.Right)
        {
            righthandcomplete.HideHandOnSelectEMG();
        }
        
        if(CurrentHand == Hand.Left)
        {
            lefthandcomplete.HideHandOnSelectEMG();
        }
    
    
    }

}